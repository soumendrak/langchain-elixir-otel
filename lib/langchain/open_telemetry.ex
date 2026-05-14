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
  | `llm.call` | `langchain.llm.call` | model, provider, message_count, tools_count |
  | `chain.execute` | `langchain.chain.execute` | chain_type, chain_id |
  | `tool.call` | `langchain.tool.call` | tool_name, tool_call_id |
  | `message.process` | `langchain.message.process` | message_type, role |
  """

  require OpenTelemetry.Tracer

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
    attrs = %{
      "langchain.model" => metadata[:model] || "unknown",
      "langchain.provider" => metadata[:provider] || "unknown",
      "langchain.message_count" => metadata[:message_count] || 0,
      "langchain.tools_count" => metadata[:tools_count] || 0
    }

    span_name = metadata[:span_name] || "langchain.llm.call"
    create_and_push_span(span_name, attrs)
    :ok
  end

  def handle_event([:langchain, :llm, :call, :stop], _measurements, metadata, _config) do
    span_ctx = pop_span()

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

  def handle_event([:langchain, :llm, :call, :exception], _measurements, metadata, _config) do
    span_ctx = pop_span()

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

      OpenTelemetry.Span.set_attributes(span_ctx.span, %{
        "langchain.prompt" => String.slice(prompt_text || "", 0, 8000)
      })
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

      OpenTelemetry.Span.set_attributes(span_ctx.span, %{
        "langchain.response" => String.slice(response_text || "", 0, 8000)
      })
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
    span_ctx = pop_span()

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
    span_ctx = pop_span()

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
    span_ctx = pop_span()

    if span_ctx do
      OpenTelemetry.Span.set_status(span_ctx.span, :ok)
      OpenTelemetry.Span.end_span(span_ctx.span)
    end

    :ok
  end

  def handle_event([:langchain, :message, :process, :exception], _measurements, metadata, _config) do
    span_ctx = pop_span()

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
    span_ctx = pop_span()

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
    span_ctx = pop_span()

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
    span_ctx = pop_span()

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
    span_ctx = pop_span()

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

  # ── Span Stack Management (process-dictionary based) ─────────────

  @span_stack_key :langchain_otel_span_stack

  defp create_and_push_span(name, attributes) do
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

  defp pop_span do
    stack = Process.get(@span_stack_key, [])

    case stack do
      [top | rest] ->
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
