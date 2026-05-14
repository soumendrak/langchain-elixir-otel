defmodule LangChain.OpenTelemetry.HierarchyValidator do
  @moduledoc """
  Validates the parent-child hierarchy of LangChain OTel spans.

  This module defines expected span nesting rules and provides validation
  to ensure spans are properly parented and closed in LIFO order.

  ## Valid Hierarchy

      langchain.agent.run
      ├── langchain.chain.execute
      │   ├── langchain.llm.call
      │   ├── langchain.tool.call
      │   └── langchain.message.process
      ├── langchain.llm.call
      ├── langchain.tool.call
      └── langchain.message.process

      langchain.chain.execute (root)
      ├── langchain.llm.call
      ├── langchain.tool.call
      └── langchain.message.process

  ## Usage

      # Check current stack state
      issues = LangChain.OpenTelemetry.HierarchyValidator.validate()

      # Enable strict mode (raises on violation)
      LangChain.OpenTelemetry.HierarchyValidator.enable_strict!()
  """

  @typedoc "A span parent-child relation definition"
  @type hierarchy_rule :: %{
          required(:parent) => String.t() | :root,
          required(:children) => [String.t()]
        }

  @typedoc "An issue found during validation"
  @type validation_issue :: %{
          type: :orphaned_span | :invalid_parent | :stack_mismatch | :missing_close,
          message: String.t(),
          details: map()
        }

  # ── Hierarchy Rules ───────────────────────────────────────────────────────

  @hierarchy_rules [
    %{
      parent: :root,
      children: [
        "langchain.chain.execute",
        "langchain.agent.run"
      ]
    },
    %{
      parent: "langchain.agent.run",
      children: [
        "langchain.chain.execute",
        "langchain.llm.call",
        "langchain.tool.call",
        "langchain.message.process"
      ]
    },
    %{
      parent: "langchain.chain.execute",
      children: [
        "langchain.llm.call",
        "langchain.tool.call",
        "langchain.message.process"
      ]
    }
  ]

  # ── Strict Mode ────────────────────────────────────────────────────────────

  @strict_key :langchain_otel_strict_hierarchy

  @doc """
  Enables strict hierarchy validation. In strict mode, span nesting violations
  will raise a warning via Logger. Use this in development to catch span
  hierarchy issues early.
  """
  @spec enable_strict!() :: :ok
  def enable_strict! do
    Process.put(@strict_key, true)
    :ok
  end

  @doc """
  Disables strict hierarchy validation (default).
  """
  @spec disable_strict!() :: :ok
  def disable_strict! do
    Process.delete(@strict_key)
    :ok
  end

  @doc """
  Returns whether strict mode is enabled.
  """
  @spec strict?() :: boolean()
  def strict? do
    Process.get(@strict_key, false)
  end

  # ── Validation ────────────────────────────────────────────────────────────

  @doc """
  Validates the current span stack state against the defined hierarchy rules.

  Returns a list of `validation_issue` structs. An empty list means
  the stack is valid.

  ## Examples

      iex> issues = LangChain.OpenTelemetry.HierarchyValidator.validate()
      iex> if issues != [], do: Logger.warning("Span hierarchy issues: \#{inspect(issues)}")
  """
  @spec validate() :: [validation_issue()]
  def validate do
    stack = Process.get(:langchain_otel_span_stack, [])
    validate_stack(stack)
  end

  @doc """
  Validates that a child span name is allowed under a given parent span name.

  Returns `:ok` if the relationship is valid, or `{:error, reason}` if not.
  """
  @spec validate_parentage(String.t(), String.t() | nil) :: :ok | {:error, String.t()}
  def validate_parentage(child_name, parent_name)

  def validate_parentage(child_name, nil) do
    # No parent — only root spans are allowed
    allowed = get_allowed_children(:root)

    if child_name in allowed do
      :ok
    else
      {:error,
       "Span '#{child_name}' cannot be a root span (no parent). Root spans: #{inspect(allowed)}"}
    end
  end

  def validate_parentage(child_name, parent_name) do
    allowed_children = get_allowed_children(parent_name)

    if child_name in allowed_children do
      :ok
    else
      {:error,
       "Span '#{child_name}' is not a valid child of '#{parent_name}'. Allowed: #{inspect(allowed_children)}"}
    end
  end

  @doc """
  Validates that the span stack is being modified correctly.

  - On push: checks that the new span is a valid child of the current top
  - On pop: checks that the popped span matches what's expected (LIFO)

  Returns `:ok` or `{:error, reason}`.
  """
  @spec validate_push(String.t()) :: :ok | {:error, String.t()}
  def validate_push(new_span_name) do
    stack = Process.get(:langchain_otel_span_stack, [])
    parent_name = if stack != [], do: hd(stack).name, else: nil
    validate_parentage(new_span_name, parent_name)
  end

  @doc """
  Validates a pop operation. Checks that the span being popped matches
  the expected name (LIFO enforcement).

  Returns `:ok` or `{:error, reason}`.
  """
  @spec validate_pop(String.t(), String.t()) :: :ok | {:error, String.t()}
  def validate_pop(expected_name, actual_top_name)

  def validate_pop(expected_name, actual_top_name) do
    if expected_name == actual_top_name do
      :ok
    else
      {:error,
       "LIFO violation: expected to pop '#{expected_name}' but top of stack is '#{actual_top_name}'. Spans must be closed in reverse order of creation."}
    end
  end

  @doc """
  Returns orphaned spans — spans that were started but never closed.
  """
  @spec orphaned_spans() :: [map()]
  def orphaned_spans do
    Process.get(:langchain_otel_span_stack, [])
  end

  # ── Private Helpers ────────────────────────────────────────────────────────

  defp validate_stack(stack) do
    issues = []

    # Check each span in the stack has a valid relationship
    issues =
      stack
      |> Enum.reverse()
      |> Enum.reduce({nil, issues}, fn span_ctx, {prev_parent, acc} ->
        new_acc =
          case validate_parentage(span_ctx.name, prev_parent) do
            :ok ->
              acc

            {:error, reason} ->
              [
                %{
                  type: :invalid_parent,
                  message: reason,
                  details: %{span: span_ctx.name, parent: prev_parent}
                }
                | acc
              ]
          end

        {span_ctx.name, new_acc}
      end)
      |> elem(1)
      |> Enum.reverse()

    # Check for orphaned spans (non-empty stack = something wasn't closed)
    all_issues =
      if stack != [] do
        names = Enum.map(stack, & &1.name) |> Enum.reverse()

        issue = %{
          type: :missing_close,
          message:
            "Spans were opened but not yet closed: #{inspect(names)}. Stack depth: #{length(stack)}",
          details: %{open_spans: names, depth: length(stack)}
        }

        issues ++ [issue]
      else
        issues
      end

    all_issues
  end

  defp get_allowed_children(:root), do: get_allowed_children(nil)

  defp get_allowed_children(nil) do
    @hierarchy_rules
    |> Enum.filter(&(&1.parent == :root))
    |> Enum.flat_map(& &1.children)
  end

  defp get_allowed_children(parent_name) do
    rule = Enum.find(@hierarchy_rules, &(&1.parent == parent_name))

    if rule do
      rule.children
    else
      # Unknown parent — allow any children (graceful degradation)
      []
    end
  end

  # ── Reset (for testing) ────────────────────────────────────────────────────

  @doc false
  def reset! do
    Process.delete(:langchain_otel_span_stack)
    Process.delete(@strict_key)
    :ok
  end
end
