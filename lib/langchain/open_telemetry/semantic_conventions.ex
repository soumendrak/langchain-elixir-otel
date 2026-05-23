defmodule LangChain.OpenTelemetry.SemanticConventions do
  @moduledoc """
  Maps LangChain telemetry metadata to OTel Semantic Conventions for GenAI
  (`gen_ai.*`) attributes.

  ## Reference

  OTel Semantic Conventions for GenAI:
  https://opentelemetry.io/docs/specs/semconv/gen-ai/

  ## Key Attributes

  | Attribute | Source | Description |
  |---|---|---|
  | `gen_ai.system` | Model name inference | AI system/backend (e.g. "openai", "anthropic") |
  | `gen_ai.request.model` | Telemetry metadata | Model name (e.g. "gpt-4", "claude-3-opus") |
  | `gen_ai.response.id` | Response data | Unique response identifier |
  | `gen_ai.response.finish_reasons` | Response data | Finish reasons (e.g. ["stop", "length"]) |
  | `gen_ai.usage.prompt_tokens` | Token usage struct | Input/prompt tokens |
  | `gen_ai.usage.completion_tokens` | Token usage struct | Output/completion tokens |
  | `gen_ai.usage.total_tokens` | Token usage struct | Total tokens consumed |
  """

  @doc """
  Derives `gen_ai.system` from a model name string.

  Maps known model prefixes to their provider systems:
    - `gpt-*`, `o1-*`, `o3-*`, `dall-e-*`, `tts-*`, `whisper-*` → openai
    - `claude-*` → anthropic
    - `gemini-*`, `text-bison-*`, `chat-bison-*`, `palm-*` → google
    - `mistral-*`, `open-mistral-*`, `codestral-*` → mistral
    - `llama-*`, `llama2-*`, `llama3-*` → meta
    - `command-*`, `command-r-*` → cohere
    - `deepseek-*` → deepseek
    - `perplexity-*`, `sonar-*`, `pplx-*` → perplexity
    - `groq-*` → groq
    - `grok-*` → xai
  """
  @spec infer_system(String.t()) :: String.t()
  def infer_system(model) when is_binary(model) do
    model = String.downcase(model)

    cond do
      String.starts_with?(model, ["gpt-", "o1-", "o3-", "dall-e-", "tts-", "whisper-"]) -> "openai"
      String.starts_with?(model, "claude-") -> "anthropic"
      String.starts_with?(model, ["gemini-", "text-bison-", "chat-bison-", "palm-"]) -> "google"
      String.starts_with?(model, ["mistral-", "open-mistral-", "codestral-", "pixtral-"]) -> "mistral"
      String.starts_with?(model, ["llama", "llama2", "llama3"]) -> "meta"
      String.starts_with?(model, ["command-", "command-r-"]) -> "cohere"
      String.starts_with?(model, "deepseek-") -> "deepseek"
      String.starts_with?(model, ["perplexity-", "sonar-", "pplx-"]) -> "perplexity"
      String.starts_with?(model, "groq-") -> "groq"
      String.starts_with?(model, "grok-") -> "xai"
      String.starts_with?(model, ["amazon.", "ai21.", "anthropic.", "cohere.", "meta."]) -> "aws"
      true -> "unknown"
    end
  end

  def infer_system(_), do: "unknown"

  @doc false
  @spec infer_system(map()) :: String.t()
  def infer_system(%{model: model}), do: infer_system(model)
  def infer_system(%{"model" => model}), do: infer_system(model)
  def infer_system(_), do: "unknown"

  @doc """
  Returns `gen_ai.request.*` attributes extracted from LLM call metadata.
  """
  @spec request_attributes(map()) :: %{optional(String.t()) => any()}
  def request_attributes(metadata) when is_map(metadata) do
    model = metadata[:model] || metadata["model"]

    attrs = %{}

    attrs =
      if model do
        Map.put(attrs, "gen_ai.request.model", model)
      else
        attrs
      end

    system = metadata[:system] || infer_system(metadata)

    attrs =
      if system do
        Map.put(attrs, "gen_ai.system", system)
      else
        attrs
      end

    attrs
  end

  @doc """
  Returns `gen_ai.usage.*` attributes extracted from a `TokenUsage` struct
  or usage metadata map.
  """
  @spec usage_attributes(map() | struct() | nil) :: %{optional(String.t()) => integer()}
  def usage_attributes(nil), do: %{}

  def usage_attributes(%{input: input, output: output, raw: raw} = usage)
      when is_struct(usage) do
    attrs = %{}

    attrs =
      if is_integer(input) and input >= 0 do
        Map.put(attrs, "gen_ai.usage.prompt_tokens", input)
      else
        attrs
      end

    attrs =
      if is_integer(output) and output >= 0 do
        Map.put(attrs, "gen_ai.usage.completion_tokens", output)
      else
        attrs
      end

    attrs =
      if is_integer(input) and is_integer(output) do
        Map.put(attrs, "gen_ai.usage.total_tokens", input + output)
      else
        attrs
      end

    # If raw contains total_tokens, use it as authoritative
    attrs =
      case raw do
        %{"total_tokens" => total} when is_integer(total) ->
          Map.put(attrs, "gen_ai.usage.total_tokens", total)

        _ ->
          attrs
      end

    attrs
  end

  def usage_attributes(%{} = metadata) do
    attrs = %{}

    attrs =
      case metadata[:input] || metadata["input"] do
        val when is_integer(val) and val >= 0 ->
          Map.put(attrs, "gen_ai.usage.prompt_tokens", val)

        _ ->
          attrs
      end

    attrs =
      case metadata[:output] || metadata["output"] do
        val when is_integer(val) and val >= 0 ->
          Map.put(attrs, "gen_ai.usage.completion_tokens", val)

        _ ->
          attrs
      end

    attrs =
      case metadata[:total_tokens] || metadata["total_tokens"] do
        val when is_integer(val) and val >= 0 ->
          Map.put(attrs, "gen_ai.usage.total_tokens", val)

        _ ->
          attrs
      end

    attrs
  end

  @doc """
  Returns `gen_ai.response.*` attributes extracted from response metadata.
  """
  @spec response_attributes(map()) :: %{optional(String.t()) => any()}
  def response_attributes(metadata) when is_map(metadata) do
    attrs = %{}

    attrs =
      case metadata[:response_id] || metadata["response_id"] do
        id when is_binary(id) and id != "" ->
          Map.put(attrs, "gen_ai.response.id", id)

        _ ->
          attrs
      end

    attrs =
      case metadata[:finish_reasons] || metadata["finish_reasons"] do
        reasons when is_list(reasons) and reasons != [] ->
          Map.put(attrs, "gen_ai.response.finish_reasons", reasons)

        reason when is_binary(reason) and reason != "" ->
          Map.put(attrs, "gen_ai.response.finish_reasons", [reason])

        _ ->
          attrs
      end

    attrs
  end

  @doc """
  Returns the OTel span name for an LLM call following the semantic convention:

      gen_ai.{system}.request

  Falls back to `langchain.llm.call` when the system cannot be determined.
  """
  @spec span_name(map()) :: String.t()
  def span_name(metadata) when is_map(metadata) do
    system = metadata[:system] || metadata["system"] || infer_system(metadata)

    if system && system != "unknown" do
      "gen_ai.#{system}.request"
    else
      "langchain.llm.call"
    end
  end
end
