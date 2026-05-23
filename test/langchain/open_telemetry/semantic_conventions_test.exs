defmodule LangChain.OpenTelemetry.SemanticConventionsTest do
  use ExUnit.Case, async: true

  alias LangChain.OpenTelemetry.SemanticConventions
  alias LangChain.TokenUsage

  describe "infer_system/1" do
    test "returns openai for gpt models" do
      assert SemanticConventions.infer_system("gpt-4") == "openai"
      assert SemanticConventions.infer_system("gpt-3.5-turbo") == "openai"
      assert SemanticConventions.infer_system("gpt-4-turbo") == "openai"
    end

    test "returns openai for o1 and o3 models" do
      assert SemanticConventions.infer_system("o1-preview") == "openai"
      assert SemanticConventions.infer_system("o3-mini") == "openai"
    end

    test "returns anthropic for claude models" do
      assert SemanticConventions.infer_system("claude-3-opus-20240229") == "anthropic"
      assert SemanticConventions.infer_system("claude-3-sonnet") == "anthropic"
      assert SemanticConventions.infer_system("claude-2") == "anthropic"
    end

    test "returns google for gemini models" do
      assert SemanticConventions.infer_system("gemini-1.5-pro") == "google"
      assert SemanticConventions.infer_system("gemini-1.0-pro") == "google"
    end

    test "returns mistral for mistral models" do
      assert SemanticConventions.infer_system("mistral-large-latest") == "mistral"
      assert SemanticConventions.infer_system("open-mistral-7b") == "mistral"
      assert SemanticConventions.infer_system("codestral-latest") == "mistral"
    end

    test "returns meta for llama models" do
      assert SemanticConventions.infer_system("llama3-70b") == "meta"
      assert SemanticConventions.infer_system("llama-2-7b") == "meta"
    end

    test "returns deepseek for deepseek models" do
      assert SemanticConventions.infer_system("deepseek-chat") == "deepseek"
      assert SemanticConventions.infer_system("deepseek-coder") == "deepseek"
    end

    test "returns perplexity for perplexity/sonar models" do
      assert SemanticConventions.infer_system("perplexity-online") == "perplexity"
      assert SemanticConventions.infer_system("sonar-small-chat") == "perplexity"
      assert SemanticConventions.infer_system("pplx-7b-online") == "perplexity"
    end

    test "returns xai for grok models" do
      assert SemanticConventions.infer_system("grok-1") == "xai"
      assert SemanticConventions.infer_system("grok-2") == "xai"
    end

    test "returns unknown for unrecognized models" do
      assert SemanticConventions.infer_system("custom-model-v1") == "unknown"
      assert SemanticConventions.infer_system("") == "unknown"
    end

    test "returns unknown for nil" do
      assert SemanticConventions.infer_system(nil) == "unknown"
    end

    test "handles model key from metadata map" do
      assert SemanticConventions.infer_system(%{model: "gpt-4"}) == "openai"
      assert SemanticConventions.infer_system(%{"model" => "claude-3"}) == "anthropic"
      assert SemanticConventions.infer_system(%{}) == "unknown"
    end
  end

  describe "request_attributes/1" do
    test "extracts gen_ai.request.model from metadata" do
      attrs = SemanticConventions.request_attributes(%{model: "gpt-4"})
      assert attrs["gen_ai.request.model"] == "gpt-4"
    end

    test "extracts gen_ai.system from model name" do
      attrs = SemanticConventions.request_attributes(%{model: "claude-3-opus"})
      assert attrs["gen_ai.system"] == "anthropic"
    end

    test "allows explicit system override" do
      attrs = SemanticConventions.request_attributes(%{model: "my-model", system: "custom"})
      assert attrs["gen_ai.system"] == "custom"
    end

    test "returns empty map for empty metadata" do
      assert SemanticConventions.request_attributes(%{}) == %{}
    end
  end

  describe "usage_attributes/1" do
    test "extracts from TokenUsage struct" do
      usage = TokenUsage.new!(%{input: 10, output: 20})
      attrs = SemanticConventions.usage_attributes(usage)

      assert attrs["gen_ai.usage.prompt_tokens"] == 10
      assert attrs["gen_ai.usage.completion_tokens"] == 20
      assert attrs["gen_ai.usage.total_tokens"] == 30
    end

    test "uses raw total_tokens when available" do
      usage = TokenUsage.new!(%{input: 10, output: 20, raw: %{"total_tokens" => 35}})
      attrs = SemanticConventions.usage_attributes(usage)

      # raw total_tokens takes precedence over input + output
      assert attrs["gen_ai.usage.total_tokens"] == 35
    end

    test "returns empty map for nil" do
      assert SemanticConventions.usage_attributes(nil) == %{}
    end

    test "returns empty map for empty usage" do
      usage = TokenUsage.new!(%{input: 0, output: 0})
      attrs = SemanticConventions.usage_attributes(usage)

      assert attrs["gen_ai.usage.prompt_tokens"] == 0
      assert attrs["gen_ai.usage.completion_tokens"] == 0
      assert attrs["gen_ai.usage.total_tokens"] == 0
    end
  end

  describe "response_attributes/1" do
    test "extracts response.id" do
      attrs = SemanticConventions.response_attributes(%{response_id: "resp_123"})
      assert attrs["gen_ai.response.id"] == "resp_123"
    end

    test "extracts finish_reasons from list" do
      attrs = SemanticConventions.response_attributes(%{finish_reasons: ["stop"]})
      assert attrs["gen_ai.response.finish_reasons"] == ["stop"]
    end

    test "wraps single string finish_reason in list" do
      attrs = SemanticConventions.response_attributes(%{finish_reasons: "stop"})
      assert attrs["gen_ai.response.finish_reasons"] == ["stop"]
    end

    test "returns empty map for empty metadata" do
      assert SemanticConventions.response_attributes(%{}) == %{}
    end
  end

  describe "span_name/1" do
    test "returns gen_ai.openai.request for OpenAI models" do
      assert SemanticConventions.span_name(%{model: "gpt-4"}) == "gen_ai.openai.request"
    end

    test "returns gen_ai.anthropic.request for Anthropic models" do
      assert SemanticConventions.span_name(%{model: "claude-3-opus"}) == "gen_ai.anthropic.request"
    end

    test "uses explicit system override" do
      assert SemanticConventions.span_name(%{model: "my-model", system: "custom"}) ==
               "gen_ai.custom.request"
    end

    test "falls back to langchain.llm.call for unknown models" do
      assert SemanticConventions.span_name(%{}) == "langchain.llm.call"
    end

    test "falls back to langchain.llm.call for unknown system" do
      assert SemanticConventions.span_name(%{model: "unknown-model"}) == "langchain.llm.call"
    end
  end
end
