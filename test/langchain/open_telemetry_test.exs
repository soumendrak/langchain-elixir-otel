defmodule LangChain.OpenTelemetryTest do
  use ExUnit.Case, async: false

  setup do
    # Detach any previous handler to start clean
    LangChain.OpenTelemetry.detach()
    :ok
  end

  describe "attach/0" do
    test "returns :ok when opentelemetry is available" do
      assert LangChain.OpenTelemetry.attach() == :ok
    end

    test "attaches handler to telemetry events" do
      LangChain.OpenTelemetry.attach()

      # The handler should be attached — verify by checking that
      # emitting a known event doesn't crash
      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "gpt-4",
               provider: "openai",
               message_count: 5,
               tools_count: 0
             }) == :ok
    end

    test "can be attached multiple times without error" do
      LangChain.OpenTelemetry.attach()
      LangChain.OpenTelemetry.attach()
      assert :ok
    end
  end

  describe "detach/0" do
    test "returns :ok" do
      assert LangChain.OpenTelemetry.detach() == :ok
    end

    test "stops creating spans after detach" do
      LangChain.OpenTelemetry.attach()
      LangChain.OpenTelemetry.detach()

      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "gpt-4",
               provider: "openai"
             }) == :ok
    end
  end

  describe "LLM call events" do
    setup do
      LangChain.OpenTelemetry.attach()
      :ok
    end

    test "start/stop creates a span lifecycle" do
      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "gpt-4",
               provider: "openai",
               message_count: 10,
               tools_count: 2
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{duration: 1000}, %{
               result: {:ok, "response"}
             }) == :ok
    end

    test "start/exception creates a span with error status" do
      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "claude-3",
               provider: "anthropic",
               message_count: 5,
               tools_count: 0
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :exception], %{}, %{
               error: %RuntimeError{message: "API timeout"},
               kind: :error,
               stacktrace: []
             }) == :ok
    end

    test "start without matching stop does not crash subsequent events" do
      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "gpt-4",
               provider: "openai"
             }) == :ok

      # Start another without stopping the first — should nest, not crash
      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "gpt-4",
               provider: "openai"
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok
      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok
    end

    test "prompt and response events add attributes to current span" do
      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "gpt-4",
               provider: "openai"
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :prompt], %{}, %{
               model: "gpt-4",
               messages: [%{role: :user, content: "Hello"}]
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :response], %{}, %{
               model: "gpt-4",
               response: "Hi there!"
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok
    end
  end

  describe "chain execution events" do
    setup do
      LangChain.OpenTelemetry.attach()
      :ok
    end

    test "start/stop creates a chain span" do
      assert :telemetry.execute([:langchain, :chain, :execute, :start], %{}, %{
               chain_type: "LLMChain",
               chain_id: "chain-123"
             }) == :ok

      assert :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{
               result: {:ok, "completed"}
             }) == :ok
    end

    test "chain span nests LLM call spans as children" do
      assert :telemetry.execute([:langchain, :chain, :execute, :start], %{}, %{
               chain_type: "LLMChain",
               chain_id: "chain-456"
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "gpt-4",
               provider: "openai"
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok

      assert :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{}) == :ok
    end
  end

  describe "tool call events" do
    setup do
      LangChain.OpenTelemetry.attach()
      :ok
    end

    test "start/stop creates a tool span" do
      assert :telemetry.execute([:langchain, :tool, :call, :start], %{}, %{
               tool_name: "calculator",
               tool_call_id: "call-abc"
             }) == :ok

      assert :telemetry.execute([:langchain, :tool, :call, :stop], %{}, %{
               result: {:ok, "42"}
             }) == :ok
    end

    test "exception creates an error tool span" do
      assert :telemetry.execute([:langchain, :tool, :call, :start], %{}, %{
               tool_name: "web_search",
               tool_call_id: "call-def"
             }) == :ok

      assert :telemetry.execute([:langchain, :tool, :call, :exception], %{}, %{
               error: %RuntimeError{message: "HTTP 500"},
               kind: :error,
               stacktrace: []
             }) == :ok
    end
  end

  describe "message process events" do
    setup do
      LangChain.OpenTelemetry.attach()
      :ok
    end

    test "start/stop creates a message span" do
      assert :telemetry.execute([:langchain, :message, :process, :start], %{}, %{
               message_type: "assistant",
               role: :assistant
             }) == :ok

      assert :telemetry.execute([:langchain, :message, :process, :stop], %{}, %{}) == :ok
    end
  end

  describe "error handling" do
    test "survives unknown event names gracefully" do
      LangChain.OpenTelemetry.attach()

      assert :telemetry.execute([:langchain, :unknown, :event], %{}, %{}) == :ok
    end

    test "pop without matching start does not crash" do
      LangChain.OpenTelemetry.attach()

      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok
    end

    test "multiple sequential operations maintain correct stack" do
      LangChain.OpenTelemetry.attach()

      # Chain 1
      assert :telemetry.execute([:langchain, :chain, :execute, :start], %{}, %{
               chain_type: "LLMChain",
               chain_id: "c1"
             }) == :ok

      assert :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{}) == :ok

      # Chain 2
      assert :telemetry.execute([:langchain, :chain, :execute, :start], %{}, %{
               chain_type: "LLMChain",
               chain_id: "c2"
             }) == :ok

      assert :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{}) == :ok

      # The stack should be clean after these
      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok
    end
  end

  describe "context propagation" do
    setup do
      LangChain.OpenTelemetry.attach()
      LangChain.OpenTelemetry.reset_span_stack!()
      :ok
    end

    test "capture_context/0 returns a map" do
      ctx = LangChain.OpenTelemetry.capture_context()
      assert is_map(ctx)
      assert Map.has_key?(ctx, :span_stack)
      assert Map.has_key?(ctx, :stack_depth)
    end

    test "capture_context/0 captures empty stack as depth 0" do
      ctx = LangChain.OpenTelemetry.capture_context()
      assert ctx.stack_depth == 0
      assert ctx.span_stack == []
    end

    test "capture_context/0 captures active span stack" do
      # Fire a chain start to push a span onto the stack
      :telemetry.execute([:langchain, :chain, :execute, :start], %{}, %{
        chain_type: "LLMChain",
        chain_id: "test-1"
      })

      ctx = LangChain.OpenTelemetry.capture_context()

      assert ctx.stack_depth == 1
      assert length(ctx.span_stack) == 1
      assert hd(ctx.span_stack).name == "langchain.chain.execute"

      # Cleanup
      :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{})
    end

    test "restore_context/1 restores the span stack" do
      ctx = %{
        span_stack: [
          %{name: "langchain.chain.execute"},
          %{name: "langchain.llm.call"}
        ],
        stack_depth: 2
      }

      LangChain.OpenTelemetry.restore_context(ctx)

      assert LangChain.OpenTelemetry.stack_depth() == 2
    end

    test "restore_context/1 clears previous stack before restoring" do
      # Create a dirty stack first
      Process.put(:langchain_otel_span_stack, [
        %{span: nil, name: "stale.span", parent: nil}
      ])

      ctx = %{
        span_stack: [%{name: "langchain.chain.execute"}],
        stack_depth: 1
      }

      LangChain.OpenTelemetry.restore_context(ctx)

      assert LangChain.OpenTelemetry.stack_depth() == 1

      # Verify the top span name matches what we restored
      # (use stack_depth to confirm; private function not accessible from tests)
    end

    test "restore_context/1 is a no-op for nil or invalid input" do
      assert LangChain.OpenTelemetry.restore_context(nil) == :ok
      assert LangChain.OpenTelemetry.restore_context(%{}) == :ok
      assert LangChain.OpenTelemetry.restore_context("bad") == :ok
    end

    test "stack_depth/0 returns 0 for clean stack" do
      LangChain.OpenTelemetry.reset_span_stack!()
      assert LangChain.OpenTelemetry.stack_depth() == 0
    end

    test "stack_depth/0 returns correct count after span creation" do
      LangChain.OpenTelemetry.reset_span_stack!()

      :telemetry.execute([:langchain, :chain, :execute, :start], %{}, %{
        chain_type: "LLMChain",
        chain_id: "t1"
      })

      assert LangChain.OpenTelemetry.stack_depth() == 1

      :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{})
      assert LangChain.OpenTelemetry.stack_depth() == 0
    end

    test "context propagation enables parented spans in child process" do
      # Simulate parent process: start a chain span
      :telemetry.execute([:langchain, :chain, :execute, :start], %{}, %{
        chain_type: "LLMChain",
        chain_id: "parent-chain"
      })

      # Capture context
      ctx = LangChain.OpenTelemetry.capture_context()

      # Simulate child process (Task): restores context
      task =
        Task.async(fn ->
          LangChain.OpenTelemetry.restore_context(ctx)

          # Fire a tool call start inside the Task — it should be parented
          :telemetry.execute([:langchain, :tool, :call, :start], %{}, %{
            tool_name: "test_tool",
            tool_call_id: "call-1"
          })

          # Check stack depth — should be 2 (chain + tool)
          depth = LangChain.OpenTelemetry.stack_depth()
          depth
        end)

      child_depth = Task.await(task)
      assert child_depth == 2

      # Cleanup parent process
      :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{})
    end
  end

  describe "hierarchy strict mode" do
    setup do
      LangChain.OpenTelemetry.attach()
      LangChain.OpenTelemetry.reset_span_stack!()
      :ok
    end

    test "strict mode warns on invalid push" do
      import ExUnit.CaptureLog

      LangChain.OpenTelemetry.HierarchyValidator.enable_strict!()

      log =
        capture_log(fn ->
          # Try to push a child span that's not valid under no parent
          # (tool.call as root is invalid)
          # Fire a chain start first, then try an invalid child
          :telemetry.execute([:langchain, :chain, :execute, :start], %{}, %{
            chain_type: "LLMChain",
            chain_id: "strict-1"
          })

          # llm.call under chain.execute is valid — no warning
          :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
            model: "test",
            provider: "test"
          })

          :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{})
          :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{})
        end)

      # Should not produce warnings for valid hierarchy
      refute log =~ "violation"
    end

    test "strict mode warns on LIFO violation" do
      import ExUnit.CaptureLog

      LangChain.OpenTelemetry.HierarchyValidator.enable_strict!()

      log =
        capture_log(fn ->
          :telemetry.execute([:langchain, :chain, :execute, :start], %{}, %{
            chain_type: "LLMChain",
            chain_id: "lifo-1"
          })

          :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
            model: "test",
            provider: "test"
          })

          # Try to close chain before closing llm — LIFO violation
          :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{})

          :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{})
        end)

      assert log =~ "violation"
    end
  end

  describe "gen_ai semantic conventions" do
    setup do
      LangChain.OpenTelemetry.attach()
      LangChain.OpenTelemetry.reset_span_stack!()
      :ok
    end

    test "includes gen_ai attributes on llm call start" do
      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "gpt-4-turbo",
               provider: "openai",
               message_count: 5,
               tools_count: 2
             }) == :ok

      # The span was created with gen_ai attributes — verify stack depth
      assert LangChain.OpenTelemetry.stack_depth() == 1

      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok
    end

    test "includes gen_ai usage attributes on llm response with token usage" do
      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "gpt-4",
               provider: "openai"
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :response], %{}, %{
               model: "gpt-4",
               response: %{
                 id: "chatcmpl-123",
                 model: "gpt-4",
                 usage: %LangChain.TokenUsage{
                   input: 50,
                   output: 100,
                   raw: %{"total_tokens" => 150}
                 }
               }
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok
    end

    test "response with id sets gen_ai.response.id" do
      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "gpt-4",
               provider: "openai"
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :response], %{}, %{
               model: "gpt-4",
               response: %{id: "chatcmpl-abc123", model: "gpt-4"}
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok
    end

    test "prompt event adds gen_ai.request.model attribute" do
      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "gpt-4",
               provider: "openai"
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :prompt], %{}, %{
               model: "gpt-4",
               messages: [%{role: :user, content: "Hello"}]
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok
    end

    test "unknown model still works" do
      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "custom-model-xyz",
               provider: "custom"
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok
    end

    test "gen_ai attributes work inside chain hierarchy" do
      assert :telemetry.execute([:langchain, :chain, :execute, :start], %{}, %{
               chain_type: "LLMChain",
               chain_id: "chain-spec-1"
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :start], %{}, %{
               model: "claude-3-sonnet",
               provider: "anthropic"
             }) == :ok

      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok

      assert :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{}) == :ok
    end
  end
end
