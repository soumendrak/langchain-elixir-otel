defmodule LangChain.AgentTest do
  use LangChain.BaseCase
  use Mimic

  alias LangChain.Agent
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Function
  alias LangChain.LangChainError

  setup do
    {:ok, chat} = ChatOpenAI.new(%{temperature: 0, stream: false})
    %{chat: chat}
  end

  describe "new/1" do
    test "creates an agent with required fields" do
      chat = ChatOpenAI.new!(%{temperature: 0})
      assert {:ok, agent} = Agent.new(llm: chat, task: "What is 2+2?")
      assert agent.llm == chat
      assert agent.task == "What is 2+2?"
      assert agent.tools == []
      assert agent.max_iterations == 15
      assert agent.verbose == false
      assert agent.system_prompt != nil
    end

    test "accepts keyword list" do
      chat = ChatOpenAI.new!(%{temperature: 0})
      assert {:ok, agent} = Agent.new(llm: chat, task: "Hello")
      assert agent.task == "Hello"
    end

    test "validates required fields" do
      assert {:error, errors} = Agent.new(%{})
      assert Enum.any?(errors, fn {key, _} -> key == :llm end)
      assert Enum.any?(errors, fn {key, _} -> key == :task end)
    end

    test "validates empty task" do
      chat = ChatOpenAI.new!(%{temperature: 0})
      assert {:error, errors} = Agent.new(llm: chat, task: "")
      assert Enum.any?(errors, fn {key, _} -> key == :task end)
    end

    test "new! raises on validation errors" do
      assert_raise LangChainError, fn ->
        Agent.new!(%{})
      end
    end

    test "new! works with valid attrs" do
      chat = ChatOpenAI.new!(%{temperature: 0})
      agent = Agent.new!(llm: chat, task: "Test")
      assert agent.task == "Test"
    end

    test "accepts custom max_iterations" do
      chat = ChatOpenAI.new!(%{temperature: 0})
      agent = Agent.new!(llm: chat, task: "Test", max_iterations: 5)
      assert agent.max_iterations == 5
    end

    test "accepts custom system_prompt" do
      chat = ChatOpenAI.new!(%{temperature: 0})
      agent = Agent.new!(llm: chat, task: "Test", system_prompt: "Custom prompt")
      assert agent.system_prompt == "Custom prompt"
    end
  end

  describe "build_tool_descriptions/1" do
    test "returns 'No tools available' when no tools" do
      assert Agent.build_tool_descriptions([]) == "No tools available."
    end

    test "formats tool descriptions with parameters_schema" do
      tool =
        Function.new!(%{
          name: "calculator",
          description: "Perform math",
          parameters_schema: %{
            type: "object",
            properties: %{expression: %{type: "string"}},
            required: ["expression"]
          },
          function: fn _args, _context -> {:ok, "0"} end
        })

      result = Agent.build_tool_descriptions([tool])
      assert result =~ "calculator"
      assert result =~ "Perform math"
      assert result =~ "expression"
      assert result =~ "string"
    end

    test "formats tool descriptions with parameters list" do
      tool =
        Function.new!(%{
          name: "greet",
          description: "Greet someone",
          parameters: [
            %{name: "name", type: "string", required: true}
          ],
          function: fn _args, _context -> {:ok, "Hello!"} end
        })

      result = Agent.build_tool_descriptions([tool])
      assert result =~ "greet"
      assert result =~ "name: string"
    end
  end

  describe "get_text_content/1" do
    test "extracts text from content parts" do
      msg = Message.new_assistant!("Hello world")
      assert Agent.get_text_content(msg) == "Hello world"
    end

    test "joins multiple content parts" do
      msg = %Message{
        role: :assistant,
        content: [
          ContentPart.text!("Part 1"),
          ContentPart.text!(" and Part 2")
        ]
      }

      assert Agent.get_text_content(msg) == "Part 1 and Part 2"
    end

    test "returns empty string for empty content" do
      msg = %Message{role: :assistant, content: []}
      assert Agent.get_text_content(msg) == ""
    end
  end

  describe "run/2" do
    test "returns answer for a simple question", %{chat: chat} do
      agent = Agent.new!(llm: chat, task: "What color is the sky?")

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         [
           Message.new_assistant!(
             "Thought: The sky is blue.\nFinal Answer: The sky is blue during clear weather."
           )
         ]}
      end)

      assert {:ok, result} = Agent.run(agent)
      assert result.answer == "The sky is blue during clear weather."
      assert result.run_count > 0
      assert is_list(result.steps)
      assert is_list(result.messages)
    end

    test "returns answer when no Final Answer marker exists", %{chat: chat} do
      agent = Agent.new!(llm: chat, task: "Say hello")

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Hello! How can I help you?")]}
      end)

      assert {:ok, result} = Agent.run(agent)
      assert result.answer == "Hello! How can I help you?"
    end

    test "handles single message response format (not wrapped in list)", %{chat: chat} do
      agent = Agent.new!(llm: chat, task: "Hi")

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, Message.new_assistant!("Final Answer: Hello!")}
      end)

      assert {:ok, result} = Agent.run(agent)
      assert result.answer == "Hello!"
    end

    test "handles multi-step with tool call", %{chat: chat} do
      agent =
        Agent.new!(
          llm: chat,
          task: "What is the weather?",
          tools: [
            Function.new!(%{
              name: "get_weather",
              description: "Get weather for a city",
              parameters_schema: %{
                type: "object",
                properties: %{city: %{type: "string"}},
                required: ["city"]
              },
              function: fn %{"city" => _city}, _context -> {:ok, "Sunny, 22°C"} end
            })
          ]
        )

      # First call: assistant wants to use the tool
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         [
           Message.new_assistant!(%{
             tool_calls: [
               ToolCall.new!(%{
                 call_id: "call-1",
                 name: "get_weather",
                 arguments: %{city: "Tokyo"}
               })
             ]
           })
         ]}
      end)

      # Second call: assistant has the observation and gives final answer
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         [
           Message.new_assistant!(
             "Thought: I got the weather for Tokyo. It is sunny and 22°C.\nFinal Answer: The weather in Tokyo is sunny and 22°C."
           )
         ]}
      end)

      assert {:ok, result} = Agent.run(agent)
      assert result.answer == "The weather in Tokyo is sunny and 22°C."
      assert length(result.steps) >= 1
    end

    test "extracts steps from conversation", %{chat: chat} do
      tool =
        Function.new!(%{
          name: "calculator",
          description: "Calculate expressions",
          parameters_schema: %{
            type: "object",
            properties: %{expression: %{type: "string"}},
            required: ["expression"]
          },
          function: fn %{"expression" => "2+2"}, _context -> {:ok, "4"} end
        })

      agent = Agent.new!(llm: chat, task: "What is 2+2?", tools: [tool])

      # First call: assistant thinks and decides to use the calculator tool
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         [
           Message.new_assistant!(%{
             tool_calls: [
               ToolCall.new!(%{
                 call_id: "calc-1",
                 name: "calculator",
                 arguments: %{expression: "2+2"}
               })
             ]
           })
         ]}
      end)

      # Second call: assistant got the result and gives final answer
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         [
           Message.new_assistant!(
             "Thought: The calculator returned 4.\nFinal Answer: 2+2 equals 4."
           )
         ]}
      end)

      assert {:ok, result} = Agent.run(agent)
      assert result.answer == "2+2 equals 4."
      assert length(result.steps) >= 1
    end

    test "handles error from LLMChain", %{chat: chat} do
      agent = Agent.new!(llm: chat, task: "Test")

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:error, LangChainError.exception(message: "API error")}
      end)

      assert {:error, _chain, _reason} = Agent.run(agent)
    end

    test "handles empty tools without errors", %{chat: chat} do
      agent = Agent.new!(llm: chat, task: "Say hello", tools: [])

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Final Answer: Hello!")]}
      end)

      assert {:ok, result} = Agent.run(agent)
      assert result.answer == "Hello!"
    end
  end

  describe "extract_result/3" do
    test "extracts final answer from messages" do
      messages = [
        Message.new_system!("System prompt"),
        Message.new_user!("Question"),
        Message.new_assistant!("Thought: I know this.\nFinal Answer: 42")
      ]

      chain = %{
        messages: messages,
        last_message: List.last(messages),
        custom_context: %{mode_state: %{run_count: 1}}
      }

      result = Agent.extract_result(chain, "Question", [])
      assert result.answer == "42"
    end

    test "falls back to last assistant content when no Final Answer marker" do
      messages = [
        Message.new_system!("System prompt"),
        Message.new_user!("Question"),
        Message.new_assistant!("Just a response without markers")
      ]

      chain = %{
        messages: messages,
        last_message: List.last(messages),
        custom_context: %{mode_state: %{run_count: 1}}
      }

      result = Agent.extract_result(chain, "Question", [])
      assert result.answer == "Just a response without markers"
    end

    test "returns fallback string when no assistant messages" do
      messages = [
        Message.new_system!("System prompt"),
        Message.new_user!("Question")
      ]

      chain = %{
        messages: messages,
        last_message: List.last(messages),
        custom_context: %{mode_state: %{run_count: 1}}
      }

      result = Agent.extract_result(chain, "Question", [])
      assert result.answer == "(No answer produced)"
    end

    test "extracts steps from messages with tool calls" do
      messages = [
        Message.new_system!("System prompt"),
        Message.new_user!("What is 2+2?"),
        Message.new_assistant!(
          "Thought: Let me calculate.\nAction: calculator\nAction Input: {\"expression\": \"2+2\"}"
        ),
        %Message{
          role: :tool,
          content: nil,
          tool_results: [
            %{name: "calculator", result: "4"}
          ]
        },
        Message.new_assistant!("Thought: Got 4.\nFinal Answer: 2+2 = 4")
      ]

      chain = %{
        messages: messages,
        last_message: List.last(messages),
        custom_context: %{mode_state: %{run_count: 3}}
      }

      result = Agent.extract_result(chain, "What is 2+2?", [])

      assert result.answer == "2+2 = 4"
      assert length(result.steps) >= 2
      assert result.run_count == 3

      step1 = Enum.at(result.steps, 0)
      assert step1[:thought] == "Let me calculate."
      assert step1[:action] == "calculator"
      assert step1[:observation] == "calculator: 4"

      step2 = Enum.at(result.steps, 1)
      assert step2[:thought] == "Got 4."
      assert step2[:final_answer] == "2+2 = 4"
    end
  end
end
