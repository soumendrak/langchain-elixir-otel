defmodule LangChain.Agent do
  @moduledoc """
  A ReAct (Reasoning + Acting) Agent for LangChain.

  The agent takes a task and a set of tools, then iteratively:
  1. Reasons about what to do next (Thought)
  2. Decides whether to use a tool (Action) or give the final answer
  3. Observes the tool result (Observation)
  4. Repeats until it has the final answer

  Built on top of `LangChain.Chains.LLMChain` using the `:while_needs_response` mode.

  ## Usage

      agent = LangChain.Agent.new!(%{
        llm: chat_model,
        task: "What is the weather in Tokyo?",
        tools: [weather_tool],
        max_iterations: 10
      })

      {:ok, result} = LangChain.Agent.run(agent)
      result.answer  # => "The current weather in Tokyo is 22°C and sunny."

  ## Result Structure

  The `run/2` function returns `{:ok, result}` where `result` is a map with:

    * `:answer` — The final answer string from the agent
    * `:steps` — List of maps describing each step (thought, action, observation)
    * `:messages` — Full list of messages exchanged during the run
    * `:task` — The original task
    * `:tools` — The tools provided
    * `:run_count` — Number of LLM calls made

  ## System Prompt

  The default system prompt instructs the LLM to use the ReAct format. You can
  provide a custom prompt via the `:system_prompt` option. Include
  `{tool_descriptions}` as a placeholder where tools should be listed.

      LangChain.Agent.new!(%{
        llm: model,
        task: "Translate hello to French",
        system_prompt: "You are a translator. Available tools:\n{tool_descriptions}\nAnswer directly."
      })
  """

  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Telemetry

  @default_system_prompt """
  You are a helpful AI assistant with access to tools.

  Use the following format to work through the task step by step:

  Thought: What you need to do next, step by step reasoning
  Action: The name of a tool to use
  Action Input: A JSON object with the tool's arguments

  When you receive the Observation (result of the tool), use it to continue:

  Thought: What the observation means and what to do next
  Action: Another tool name
  ...(repeat Thought/Action/Observation as needed)...

  Once you have all the information needed to answer:

  Thought: I now have all the information needed
  Final Answer: The complete answer to the user's question

  Available tools:
  {tool_descriptions}

  Remember:
  - Always think before acting
  - Use exact tool names as provided
  - Provide Action Input as valid JSON matching the tool's parameters
  - Stop and provide Final Answer once you have enough information
  """

  @type t :: %__MODULE__{
          llm: term(),
          task: String.t(),
          tools: list(Function.t()),
          system_prompt: String.t(),
          max_iterations: pos_integer(),
          verbose: boolean()
        }

  defstruct [:llm, :task, :tools, :system_prompt, :max_iterations, :verbose]

  # ── Construction ──────────────────────────────────────────────────────────

  @doc """
  Create a new Agent configuration.

  ## Options (as map or keyword list)

    * `:llm` — (required) A chat model struct implementing `ChatModel` behaviour
    * `:task` — (required) The task or question for the agent
    * `:tools` — (optional) List of `Function` structs, default: `[]`
    * `:system_prompt` — (optional) Custom system prompt override
    * `:max_iterations` — (optional) Max LLM calls, default: 15
    * `:verbose` — (optional) Enable verbose logging, default: `false`

  Returns `{:ok, agent}` or `{:error, errors}`.
  """
  @spec new(map() | keyword()) :: {:ok, t()} | {:error, term()}
  def new(attrs \\ [])

  def new(attrs) when is_list(attrs), do: new(Map.new(attrs))

  def new(%{} = attrs) do
    struct = %__MODULE__{
      llm: Map.get(attrs, :llm),
      task: Map.get(attrs, :task, ""),
      tools: Map.get(attrs, :tools, []),
      system_prompt: Map.get(attrs, :system_prompt, @default_system_prompt),
      max_iterations: Map.get(attrs, :max_iterations, 15),
      verbose: Map.get(attrs, :verbose, false)
    }

    errors =
      []
      |> then(fn err ->
        if is_nil(struct.llm),
          do: [{:llm, {"is required", [validation: :required]}} | err],
          else: err
      end)
      |> then(fn err ->
        if is_nil(struct.task) || struct.task == "",
          do: [{:task, {"is required", [validation: :required]}} | err],
          else: err
      end)

    if errors == [] do
      {:ok, struct}
    else
      {:error, errors}
    end
  end

  @doc """
  Same as `new/1` but raises on validation errors.
  """
  @spec new!(map() | keyword()) :: t() | no_return()
  def new!(attrs \\ []) do
    case new(attrs) do
      {:ok, agent} ->
        agent

      {:error, errors} ->
        raise LangChainError, message: "Agent validation failed: #{inspect(errors)}"
    end
  end

  # ── Execution ────────────────────────────────────────────────────────────

  @doc """
  Run the agent to completion.

  Builds an LLMChain with the ReAct system prompt, runs it in
  `:while_needs_response` mode, and extracts the final answer and steps.

  ## Options

  Passed through to `LLMChain.run/2`. Currently unused by the agent itself.

  ## Return

  Returns `{:ok, result}` where `result` is a map with keys `:answer`, `:steps`,
  `:messages`, `:task`, `:tools`, and `:run_count`.

  Returns `{:error, chain, reason}` if the chain execution fails.
  """
  @spec run(t(), keyword()) :: {:ok, map()} | {:error, term()}
  def run(%__MODULE__{} = agent, opts \\ []) do
    stop =
      Telemetry.start_event([:langchain, :agent, :run], %{
        task: agent.task,
        tool_count: length(agent.tools),
        max_iterations: agent.max_iterations
      })

    try do
      result = do_run(agent, opts)
      stop.(%{result: :ok})
      result
    rescue
      exception ->
        stacktrace = __STACKTRACE__

        Telemetry.emit_event(
          [:langchain, :agent, :run, :exception],
          %{system_time: System.system_time()},
          %{kind: :error, error: exception, stacktrace: stacktrace}
        )

        reraise exception, stacktrace
    catch
      kind, value ->
        Telemetry.emit_event(
          [:langchain, :agent, :run, :exception],
          %{system_time: System.system_time()},
          %{kind: kind, error: value}
        )

        :erlang.raise(kind, value, __STACKTRACE__)
    end
  end

  # ── Private ───────────────────────────────────────────────────────────────

  defp do_run(
         %__MODULE__{
           llm: llm,
           task: task,
           tools: tools,
           system_prompt: system_prompt,
           max_iterations: max_iterations,
           verbose: verbose
         } = _agent,
         _opts
       ) do
    tool_descriptions = build_tool_descriptions(tools)

    full_system_prompt =
      if String.contains?(system_prompt, "{tool_descriptions}") do
        String.replace(system_prompt, "{tool_descriptions}", tool_descriptions)
      else
        system_prompt <> "\n\nAvailable tools:\n" <> tool_descriptions
      end

    chain =
      %{llm: llm, tools: tools, verbose: verbose}
      |> LLMChain.new!()
      |> LLMChain.add_message(Message.new_system!(full_system_prompt))
      |> LLMChain.add_message(Message.new_user!(task))

    case LLMChain.run(chain, mode: :while_needs_response, max_runs: max_iterations) do
      {:ok, updated_chain} ->
        result = extract_result(updated_chain, task, tools)
        {:ok, result}

      {:error, _chain, _reason} = error ->
        error

      other ->
        {:error, nil, "Agent execution terminated: #{inspect(other)}"}
    end
  end

  # ── Tool Description Formatting ──────────────────────────────────────────

  @doc false
  def build_tool_descriptions([]), do: "No tools available."

  def build_tool_descriptions(tools) when is_list(tools) do
    tools
    |> Enum.map(fn tool ->
      params_desc = describe_params(tool)
      "- `#{tool.name}`: #{tool.description || "No description"}\n  Parameters: #{params_desc}"
    end)
    |> Enum.join("\n\n")
  end

  def build_tool_descriptions(_tools), do: "No tools available."

  defp describe_params(%Function{parameters: params}) when is_list(params) and params != [] do
    Enum.map_join(params, ", ", fn p ->
      "#{p.name}: #{p.type}#{if p.required, do: " (required)", else: ""}"
    end)
  end

  # parameters_schema uses atom keys after Function.new! processing
  defp describe_params(%Function{parameters_schema: %{properties: props}})
       when is_map(props) and map_size(props) > 0 do
    Enum.map_join(props, ", ", fn {name, schema} ->
      type = Map.get(schema, :type, "any")
      "#{name}: #{type}"
    end)
  end

  defp describe_params(_), do: "None"

  # ── Result Extraction ────────────────────────────────────────────────────

  @doc false
  def extract_result(chain, task, tools) do
    messages = chain.messages || []

    final_answer =
      messages
      |> Enum.reverse()
      |> Enum.find_value(fn msg ->
        if msg.role == :assistant do
          content = get_text_content(msg)

          case String.split(content, "Final Answer:") do
            [_, answer] -> String.trim(answer)
            _ -> nil
          end
        end
      end)

    steps = extract_steps(messages)

    %{
      answer: final_answer || extract_fallback_answer(messages),
      steps: steps,
      messages: messages,
      task: task,
      tools: tools,
      run_count: get_run_count(chain)
    }
  end

  defp extract_fallback_answer(messages) do
    messages
    |> Enum.reverse()
    |> Enum.find_value(fn msg ->
      if msg.role == :assistant do
        content = get_text_content(msg)
        if content != "", do: content, else: nil
      end
    end) || "(No answer produced)"
  end

  defp extract_steps(messages) do
    messages
    |> Enum.reduce([], fn msg, acc ->
      if msg.role == :assistant do
        content = get_text_content(msg)
        step = extract_step_from_content(content)

        if step == %{}, do: acc, else: acc ++ [step]
      else
        if msg.role == :tool do
          observation = extract_observation(msg)

          if observation do
            last = List.last(acc) || %{}
            acc = if acc != [], do: Enum.drop(acc, -1), else: acc
            acc ++ [Map.put(last, :observation, observation)]
          else
            acc
          end
        else
          acc
        end
      end
    end)
  end

  defp extract_step_from_content(content) when is_binary(content) do
    thought =
      case Regex.run(~r/Thought:(.*?)(?=Action:|Final Answer:|$)/s, content) do
        [_, t] -> String.trim(t)
        nil -> nil
      end

    action =
      case Regex.run(~r/Action:(.*?)(?=Action Input:|$)/s, content) do
        [_, a] -> String.trim(a)
        nil -> nil
      end

    action_input =
      case Regex.run(~r/Action Input:(.*?)(?=Observation:|Thought:|Final Answer:|$)/s, content) do
        [_, i] -> String.trim(i)
        nil -> nil
      end

    final_answer =
      case Regex.run(~r/Final Answer:(.*?)$/s, content) do
        [_, f] -> String.trim(f)
        nil -> nil
      end

    step = %{}
    step = if thought, do: Map.put(step, :thought, thought), else: step
    step = if action, do: Map.put(step, :action, action), else: step
    step = if action_input, do: Map.put(step, :action_input, action_input), else: step
    step = if final_answer, do: Map.put(step, :final_answer, final_answer), else: step
    step
  end

  defp extract_step_from_content(_), do: %{}

  defp extract_observation(%{tool_results: tool_results}) when is_list(tool_results) do
    results =
      Enum.map(tool_results, fn tr ->
        case tr do
          %{name: name, result: result} when is_binary(result) ->
            "#{name}: #{result}"

          %{name: name, result: result} ->
            "#{name}: #{inspect(result)}"

          _ ->
            nil
        end
      end)
      |> Enum.reject(&is_nil/1)

    if results != [], do: Enum.join(results, "\n"), else: nil
  end

  defp extract_observation(_), do: nil

  defp get_run_count(chain) do
    case chain do
      %{custom_context: %{mode_state: %{run_count: count}}} -> count
      _ -> 0
    end
  end

  # ── Content Helpers ──────────────────────────────────────────────────────

  @doc false
  def get_text_content(%Message{content: content_parts}) when is_list(content_parts) do
    content_parts
    |> Enum.map(fn
      %ContentPart{type: :text, content: text} -> text
      _ -> ""
    end)
    |> Enum.join("")
  end

  def get_text_content(_), do: ""
end
