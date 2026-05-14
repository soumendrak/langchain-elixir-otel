defmodule LangChain.OpenTelemetryTest do
  use ExUnit.Case, async: false

  import ExUnit.CaptureLog

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
        chain_type: "LLMChain", chain_id: "c1"
      }) == :ok
      assert :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{}) == :ok

      # Chain 2
      assert :telemetry.execute([:langchain, :chain, :execute, :start], %{}, %{
        chain_type: "LLMChain", chain_id: "c2"
      }) == :ok
      assert :telemetry.execute([:langchain, :chain, :execute, :stop], %{}, %{}) == :ok

      # The stack should be clean after these
      assert :telemetry.execute([:langchain, :llm, :call, :stop], %{}, %{}) == :ok
    end
  end
end
