defmodule LangChain.OpenTelemetry do
  @moduledoc """
  Provides OpenTelemetry integration for LangChain by bridging Elixir's
  `:telemetry` events to OpenTelemetry spans.

  This module attaches to the `:telemetry` events emitted by `LangChain.Telemetry`
  and creates corresponding OpenTelemetry spans with appropriate attributes
  and parent-child relationships.

  ## Dependency

  This module requires the `opentelemetry_api` package to be included as a
  dependency in your application. In LangChain, it's an optional dependency —
  the module compiles without it, but span creation is a no-op.

  ## Setup

  In your application's supervision tree, call `LangChain.OpenTelemetry.attach/0`
  to start listening for telemetry events:

      def start(_type, _args) do
        :ok = LangChain.OpenTelemetry.attach()

        children = [
          # ... your other children
        ]

        opts = [strategy: :one_for_one, name: MyApp.Supervisor]
        Supervisor.start_link(children, opts)
      end

  The `:telemetry` events from LangChain will now create OpenTelemetry spans
  that are exported by your configured OpenTelemetry exporter.

  ## Span Hierarchy

  Events are mapped to spans with the following hierarchy:

      chain.execute
      ├── llm.call
      │   ├── llm.prompt
      │   └── llm.response
      ├── message.process
      └── tool.call

  ## Event → Span Mapping

  | Telemetry Event | OTel Span Name | Attributes |
  |---|---|---|
  | `llm.call` | `langchain.llm.call` | model, provider, message_count, tools_count + gen_ai.request.model, gen_ai.system |
  | `chain.execute` | `langchain.chain.execute` | chain_type, chain_id |
  | `tool.call` | `langchain.tool.call` | tool_name, tool_call_id |
  | `message.process` | `langchain.message.process` | message_type, role |
  | `agent.run` | `langchain.agent.run` | task, tool_count, max_iterations |

  LLM call spans also receive `gen_ai.usage.*` (prompt_tokens, completion_tokens,
  total_tokens) and `gen_ai.response.id` attributes from the response event
  when the data is available.

  ## Hierarchy Validation

  Use `LangChain.OpenTelemetry.HierarchyValidator` to validate span parent-child
  relationships. Enable strict mode during development:

      LangChain.OpenTelemetry.HierarchyValidator.enable_strict!()

  ## Async Context Propagation

  When tool calls execute in separate Elixir Tasks (via `Task.async`), the
  OTel span context does not automatically propagate. Use the context
  propagation helpers:

      # In parent process — before spawning Task
      ctx = LangChain.OpenTelemetry.capture_context()

      # In the child Task process — restore the context
      LangChain.OpenTelemetry.restore_context(ctx)

  LangChain's async tool execution automatically handles this via the
  telemetry metadata propagation mechanism. If you spawn custom Tasks
  that emit LangChain telemetry events, use the helpers above.
  """

  require OpenTelemetry.Tracer

  alias LangChain.OpenTelemetry.HierarchyValidator
  alias LangChain.OpenTelemetry.SemanticConventions

  @span_stack_key :langchain_otel_span_stack

  @telemetry_events [
    [:langchain, :llm, :call, :start],
    [:langchain, :llm, :call, :stop],
    [:langchain, :llm, :call, :exception],
    [:langchain, :llm, :prompt],
    [:langchain, :llm, :response],
    [:langchain, :chain, :execute, :start],
    [:langchain, :chain, :execute, :stop],
    [:langchain, :chain, :execute, :exception],
    [:langchain, :message, :process, :start],
    [:langchain, :message, :process, :stop],
    [:langchain, :message, :process, :exception],
    [:langchain, :tool, :call, :start],
    [:langchain, :tool, :call, :stop],
    [:langchain, :tool, :call, :exception],
    [:langchain, :agent, :run, :start],
    [:langchain, :agent, :run, :stop],
    [:langchain, :agent, :run, :exception],
    [:langchain, :agent, :step]
  ]

  @doc """
  Attaches the OpenTelemetry handler to LangChain's `:telemetry` events.

  Call this once during application startup (typically in `start/2` of your
  Application module) to begin creating OpenTelemetry spans from LangChain
  operations.

  Returns `:ok` if the handler was attached, or `{:error, reason}` if
  `opentelemetry_api` is not available or attachment fails.

  ## Examples

      iex> LangChain.OpenTelemetry.attach()
      :ok

  """
  @spec attach() :: :ok | {:error, term()}
  def attach do
    if Code.ensure_loaded?(:opentelemetry) do
      :telemetry.attach_many(
        "langchain-otel-handler",
        @telemetry_events,
        &handle_event/4,
        :no_config
      )
    else
      {:error, :opentelemetry_not_available}
    end
  end

  @doc """
  Detaches the OpenTelemetry handler from LangChain's `:telemetry` events.

  Call this during application shutdown or when you need to temporarily
  disable OpenTelemetry tracing for LangChain operations.

  Returns `:ok` regardless of whether the handler was previously attached.

  ## Examples

      iex> LangChain.OpenTelemetry.detach()
      :ok

  """
  @spec detach() :: :ok
  def detach do
    :telemetry.detach("langchain-otel-handler")
    :ok
  end

  @doc false
  @spec handle_event([atom()], map(), map(), term()) :: :ok
  def handle_event(event_name, measurements, metadata, _config)

  # ── LLM Call Events ──────────────────────────────────────────────

  def handle_event([:langchain, :llm, :call, :start], _measurements, metadata, _config) do
    gen_ai_attrs = SemanticConventions.request_attributes(metadata)

    attrs =
      Map.merge(
        %{
          "langchain.model" => metadata[:model] || "unknown",
          "langchain.provider" => metadata[:provider] || "unknown",
          "langchain.message_count" => metadata[:message_count] || 0,
          "langchain.tools_count" => metadata[:tools_count] || 0
        },
        gen_ai_attrs
      )

    span_name = metadata[:span_name] || "langchain.llm.call"
    create_and_push_span(span_name, attrs)
    :ok
  end

  def handle_event([:langchain, :llm, :call, :stop], _measurements, metadata, _config) do
    span_ctx = pop_span("langchain.llm.call")

    if span_ctx do
      # Add gen_ai.usage.* attributes from the result
      case metadata[:result] do
        {:ok, result} when is_map(result) ->
          if result[:usage] do
            usage_attrs = SemanticConventions.usage_attributes(result[:usage])
            OpenTelemetry.Span.set_attributes(span_ctx.span, usage_attrs)
          end

        _ ->
          :ok
      end

      # Add gen_ai.response.* attributes if present in stop metadata
      resp_attrs = SemanticConventions.response_attributes(metadata)

      if resp_attrs != %{} do
        OpenTelemetry.Span.set_attributes(span_ctx.span, resp_attrs)
      end

      if metadata[:result] do
        status =
          case metadata[:result] do
            {:ok, _} -> :ok
            {:error, _} -> :error
            _ -> :ok
          end

        OpenTelemetry.Span.set_status(span_ctx.span, status)
      end

      OpenTelemetry.Span.end_span(span_ctx.span)
    end

    :ok
  end

  def handle_event([:langchain, :llm, :call, :exception], _measurements, metadata, _config) do
    span_ctx = pop_span("langchain.llm.call")

    if span_ctx do
      OpenTelemetry.Span.set_status(span_ctx.span, :error)

      if metadata[:error] do
        OpenTelemetry.Span.record_exception(span_ctx.span, metadata[:error],
          stack_trace: metadata[:stacktrace]
        )
      end

      OpenTelemetry.Span.end_span(span_ctx.span)
    end

    :ok
  end

  # ── LLM Prompt/Response (add attributes to current span) ─────────

  def handle_event([:langchain, :llm, :prompt], _measurements, metadata, _config) do
    span_ctx = current_span()

    if span_ctx do
      prompt_text =
        case metadata[:messages] do
          messages when is_list(messages) ->
            Enum.map_join(messages, "\n", fn m ->
              "#{m.role}: #{inspect(m.content)}"
            end)

          _ ->
            inspect(metadata[:messages])
        end

      attrs = %{
        "langchain.prompt" => String.slice(prompt_text || "", 0, 8000)
      }

      model = metadata[:model] || metadata["model"]

      attrs =
        if model do
          Map.put(attrs, "gen_ai.request.model", model)
        else
          attrs
        end

      OpenTelemetry.Span.set_attributes(span_ctx.span, attrs)
    end

    :ok
  end

  def handle_event([:langchain, :llm, :response], _measurements, metadata, _config) do
    span_ctx = current_span()

    if span_ctx do
      response_text =
        case metadata[:response] do
          response when is_binary(response) -> response
          response when is_map(response) -> inspect(response)
          _ -> inspect(metadata[:response])
        end

      attrs = %{
        "langchain.response" => String.slice(response_text || "", 0, 8000)
      }

      # Extract gen_ai.usage attributes from response data
      usage_attrs =
        case metadata[:response] do
          %{metadata: %{usage: %{input: _, output: _} = usage}} ->
            SemanticConventions.usage_attributes(usage)

          %{usage: %{input: _, output: _} = usage} ->
            SemanticConventions.usage_attributes(usage)

          _ ->
            %{}
        end

      # Extract gen_ai.response.id from response if available
      response_id =
        case metadata[:response] do
          %{id: id} when is_binary(id) -> id
          %{"id" => id} when is_binary(id) -> id
          _ -> nil
        end

      resp_attrs =
        if response_id do
          Map.put(usage_attrs, "gen_ai.response.id", response_id)
        else
          usage_attrs
        end

      all_attrs = Map.merge(attrs, resp_attrs)

      OpenTelemetry.Span.set_attributes(span_ctx.span, all_attrs)
    end

    :ok
  end

  # ── Chain Execute Events ─────────────────────────────────────────

  def handle_event([:langchain, :chain, :execute, :start], _measurements, metadata, _config) do
    attrs = %{
      "langchain.chain.type" => metadata[:chain_type] || "unknown",
      "langchain.chain.id" => metadata[:chain_id] || "unknown"
    }

    span_name = "langchain.chain.execute"
    create_and_push_span(span_name, attrs)
    :ok
  end

  def handle_event([:langchain, :chain, :execute, :stop], _measurements, metadata, _config) do
    span_ctx = pop_span("langchain.chain.execute")

    if span_ctx do
      if metadata[:result] do
        status =
          case metadata[:result] do
            {:ok, _} -> :ok
            {:error, _} -> :error
            _ -> :ok
          end

        OpenTelemetry.Span.set_status(span_ctx.span, status)
      end

      OpenTelemetry.Span.end_span(span_ctx.span)
    end

    :ok
  end

  def handle_event([:langchain, :chain, :execute, :exception], _measurements, metadata, _config) do
    span_ctx = pop_span("langchain.chain.execute")

    if span_ctx do
      OpenTelemetry.Span.set_status(span_ctx.span, :error)

      if metadata[:error] do
        OpenTelemetry.Span.record_exception(span_ctx.span, metadata[:error],
          stack_trace: metadata[:stacktrace]
        )
      end

      OpenTelemetry.Span.end_span(span_ctx.span)
    end

    :ok
  end

  # ── Message Process Events ───────────────────────────────────────

  def handle_event([:langchain, :message, :process, :start], _measurements, metadata, _config) do
    attrs = %{
      "langchain.message.type" => metadata[:message_type] || "unknown",
      "langchain.message.role" => metadata[:role] || "unknown"
    }

    span_name = "langchain.message.process"
    create_and_push_span(span_name, attrs)
    :ok
  end

  def handle_event([:langchain, :message, :process, :stop], _measurements, _metadata, _config) do
    span_ctx = pop_span("langchain.message.process")

    if span_ctx do
      OpenTelemetry.Span.set_status(span_ctx.span, :ok)
      OpenTelemetry.Span.end_span(span_ctx.span)
    end

    :ok
  end

  def handle_event([:langchain, :message, :process, :exception], _measurements, metadata, _config) do
    span_ctx = pop_span("langchain.message.process")

    if span_ctx do
      OpenTelemetry.Span.set_status(span_ctx.span, :error)

      if metadata[:error] do
        OpenTelemetry.Span.record_exception(span_ctx.span, metadata[:error],
          stack_trace: metadata[:stacktrace]
        )
      end

      OpenTelemetry.Span.end_span(span_ctx.span)
    end

    :ok
  end

  # ── Tool Call Events ─────────────────────────────────────────────

  def handle_event([:langchain, :tool, :call, :start], _measurements, metadata, _config) do
    attrs = %{
      "langchain.tool.name" => metadata[:tool_name] || "unknown",
      "langchain.tool.call_id" => metadata[:tool_call_id] || "unknown"
    }

    span_name = "langchain.tool.call"
    create_and_push_span(span_name, attrs)
    :ok
  end

  def handle_event([:langchain, :tool, :call, :stop], _measurements, metadata, _config) do
    span_ctx = pop_span("langchain.tool.call")

    if span_ctx do
      if metadata[:result] do
        status =
          case metadata[:result] do
            {:ok, _} -> :ok
            {:error, _} -> :error
            _ -> :ok
          end

        OpenTelemetry.Span.set_status(span_ctx.span, status)
      end

      OpenTelemetry.Span.end_span(span_ctx.span)
    end

    :ok
  end

  def handle_event([:langchain, :tool, :call, :exception], _measurements, metadata, _config) do
    span_ctx = pop_span("langchain.tool.call")

    if span_ctx do
      OpenTelemetry.Span.set_status(span_ctx.span, :error)

      if metadata[:error] do
        OpenTelemetry.Span.record_exception(span_ctx.span, metadata[:error],
          stack_trace: metadata[:stacktrace]
        )
      end

      OpenTelemetry.Span.end_span(span_ctx.span)
    end

    :ok
  end

  # ── Agent Run Events ──────────────────────────────────────────────

  def handle_event([:langchain, :agent, :run, :start], _measurements, metadata, _config) do
    attrs = %{
      "langchain.agent.task" => String.slice(metadata[:task] || "", 0, 200),
      "langchain.agent.tool_count" => metadata[:tool_count] || 0,
      "langchain.agent.max_iterations" => metadata[:max_iterations] || 15
    }

    span_name = "langchain.agent.run"
    create_and_push_span(span_name, attrs)
    :ok
  end

  def handle_event([:langchain, :agent, :run, :stop], _measurements, metadata, _config) do
    span_ctx = pop_span("langchain.agent.run")

    if span_ctx do
      status =
        case metadata[:result] do
          :ok -> :ok
          _ -> :error
        end

      OpenTelemetry.Span.set_status(span_ctx.span, status)
      OpenTelemetry.Span.end_span(span_ctx.span)
    end

    :ok
  end

  def handle_event([:langchain, :agent, :run, :exception], _measurements, metadata, _config) do
    span_ctx = pop_span("langchain.agent.run")

    if span_ctx do
      OpenTelemetry.Span.set_status(span_ctx.span, :error)

      if metadata[:error] do
        OpenTelemetry.Span.record_exception(span_ctx.span, metadata[:error],
          stack_trace: metadata[:stacktrace]
        )
      end

      OpenTelemetry.Span.end_span(span_ctx.span)
    end

    :ok
  end

  # ── Agent Step Events (add attribute to current span) ────────────

  def handle_event([:langchain, :agent, :step], _measurements, metadata, _config) do
    span_ctx = current_span()

    if span_ctx do
      step_type = metadata[:type] || "unknown"

      OpenTelemetry.Span.add_event(span_ctx.span, "agent.step.#{step_type}", %{
        "langchain.agent.step.type" => step_type,
        "langchain.agent.step.content" => String.slice(metadata[:content] || "", 0, 1000)
      })
    end

    :ok
  end

  # ── Context Propagation ──────────────────────────────────────────

  @doc """
  Captures the current OTel span context for propagation across process boundaries.

  Returns a serializable map containing the current span stack state.
  Use `restore_context/1` in a child process (e.g., inside a `Task.async`)
  to restore parent-child span relationships.

  ## Examples

      ctx = LangChain.OpenTelemetry.capture_context()

      Task.async(fn ->
        LangChain.OpenTelemetry.restore_context(ctx)
        # Telemetry events here will be properly parented
      end)
  """
  @spec capture_context() :: map()
  def capture_context do
    span_stack = Process.get(@span_stack_key, [])
    # Serialize only the span names — not the actual OTel structs
    serialized_stack =
      Enum.map(span_stack, fn %{name: name} -> %{name: name} end)

    %{
      span_stack: serialized_stack,
      stack_depth: length(span_stack)
    }
  end

  @doc """
  Restores a previously captured span context in the current process.

  Call this inside a child process (e.g., a `Task.async`) before
  executing code that emits LangChain telemetry events. This ensures
  that spans created in the child process are properly parented under
  the span that was active in the parent process.

  Returns `:ok`.
  """
  @spec restore_context(map()) :: :ok
  def restore_context(%{span_stack: span_stack} = _ctx) when is_list(span_stack) do
    # Clear any existing stack first to avoid contamination
    Process.delete(@span_stack_key)

    # Restore the span stack metadata
    restored =
      Enum.map(span_stack, fn %{name: name} ->
        %{name: name, span: nil, parent: nil}
      end)

    Process.put(@span_stack_key, restored)
    :ok
  end

  def restore_context(_), do: :ok

  @doc """
  Returns the current span stack depth. Useful for detecting if spans
  are being properly closed across process boundaries.

  0 means no active span context.
  """
  @spec stack_depth() :: non_neg_integer()
  def stack_depth do
    Process.get(@span_stack_key, []) |> length()
  end

  @doc false
  def reset_span_stack! do
    Process.delete(@span_stack_key)
    :ok
  end

  # ── Span Stack Management (process-dictionary based) ─────────────

  defp create_and_push_span(name, attributes) do
    # Hierarchy validation (strict mode)
    if HierarchyValidator.strict?() do
      case HierarchyValidator.validate_push(name) do
        {:error, reason} ->
          require Logger
          Logger.warning("OTel span hierarchy violation: #{reason}")

        :ok ->
          :ok
      end
    end

    parent_span = current_span()

    span = OpenTelemetry.Tracer.start_span(name, %{attributes: attributes})

    if parent_span do
      OpenTelemetry.Tracer.set_current_span(span)
    end

    # Push onto the span stack so nested events create child spans
    stack = Process.get(@span_stack_key, [])
    span_ctx = %{span: span, name: name, parent: parent_span}
    Process.put(@span_stack_key, [span_ctx | stack])

    span_ctx
  end

  defp pop_span(expected_name) do
    stack = Process.get(@span_stack_key, [])

    case stack do
      [top | rest] ->
        # Hierarchy validation (strict mode) — LIFO check
        if HierarchyValidator.strict?() && expected_name do
          case HierarchyValidator.validate_pop(expected_name, top.name) do
            {:error, reason} ->
              require Logger
              Logger.warning("OTel span hierarchy violation: #{reason}")

            :ok ->
              :ok
          end
        end

        Process.put(@span_stack_key, rest)
        top

      [] ->
        nil
    end
  end

  defp current_span do
    stack = Process.get(@span_stack_key, [])

    case stack do
      [top | _] -> top
      [] -> nil
    end
  end
end
