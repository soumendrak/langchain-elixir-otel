defmodule LangChain.OpenTelemetry.HierarchyValidatorTest do
  use ExUnit.Case, async: true

  alias LangChain.OpenTelemetry.HierarchyValidator

  setup do
    HierarchyValidator.reset!()
    HierarchyValidator.disable_strict!()
    :ok
  end

  describe "validate_parentage/2" do
    test "allows root spans (chain.execute) with no parent" do
      assert HierarchyValidator.validate_parentage("langchain.chain.execute", nil) == :ok
    end

    test "allows root spans (agent.run) with no parent" do
      assert HierarchyValidator.validate_parentage("langchain.agent.run", nil) == :ok
    end

    test "allows llm.call under chain.execute" do
      assert HierarchyValidator.validate_parentage(
               "langchain.llm.call",
               "langchain.chain.execute"
             ) == :ok
    end

    test "allows llm.call under agent.run" do
      assert HierarchyValidator.validate_parentage(
               "langchain.llm.call",
               "langchain.agent.run"
             ) == :ok
    end

    test "allows tool.call under chain.execute" do
      assert HierarchyValidator.validate_parentage(
               "langchain.tool.call",
               "langchain.chain.execute"
             ) == :ok
    end

    test "allows message.process under chain.execute" do
      assert HierarchyValidator.validate_parentage(
               "langchain.message.process",
               "langchain.chain.execute"
             ) == :ok
    end

    test "allows chain.execute under agent.run" do
      assert HierarchyValidator.validate_parentage(
               "langchain.chain.execute",
               "langchain.agent.run"
             ) == :ok
    end

    test "rejects tool.call as root span (no parent)" do
      assert {:error, _reason} =
               HierarchyValidator.validate_parentage("langchain.tool.call", nil)
    end

    test "rejects llm.call as root span (no parent)" do
      assert {:error, _reason} =
               HierarchyValidator.validate_parentage("langchain.llm.call", nil)
    end

    test "rejects unknown child under known parent" do
      assert {:error, _reason} =
               HierarchyValidator.validate_parentage(
                 "langchain.unknown.span",
                 "langchain.chain.execute"
               )
    end

    test "rejects chain.execute under llm.call (wrong nesting)" do
      # chain.execute should not be a child of llm.call
      result =
        HierarchyValidator.validate_parentage(
          "langchain.chain.execute",
          "langchain.llm.call"
        )

      # llm.call has no children defined, so it should reject
      assert {:error, _reason} = result
    end
  end

  describe "validate_push/1" do
    setup do
      # Simulate a chain.execute span on the stack
      Process.put(:langchain_otel_span_stack, [
        %{span: nil, name: "langchain.chain.execute", parent: nil}
      ])

      :ok
    end

    test "allows valid child push under chain.execute" do
      assert HierarchyValidator.validate_push("langchain.llm.call") == :ok
    end

    test "rejects invalid child push" do
      assert {:error, _} =
               HierarchyValidator.validate_push("langchain.unknown.span")
    end

    test "allows root spans when stack is empty" do
      Process.delete(:langchain_otel_span_stack)

      assert HierarchyValidator.validate_push("langchain.chain.execute") == :ok
    end
  end

  describe "validate_pop/2" do
    setup do
      Process.put(:langchain_otel_span_stack, [
        %{span: nil, name: "langchain.llm.call", parent: nil},
        %{span: nil, name: "langchain.chain.execute", parent: nil}
      ])

      :ok
    end

    test "allows correct LIFO pop (top of stack)" do
      assert HierarchyValidator.validate_pop("langchain.llm.call", "langchain.llm.call") == :ok
    end

    test "rejects wrong LIFO pop (not top of stack)" do
      assert {:error, reason} =
               HierarchyValidator.validate_pop("langchain.chain.execute", "langchain.llm.call")

      assert reason =~ "LIFO violation"
    end
  end

  describe "orphaned_spans/0" do
    test "returns empty list for clean stack" do
      assert HierarchyValidator.orphaned_spans() == []
    end

    test "returns open spans for non-empty stack" do
      Process.put(:langchain_otel_span_stack, [
        %{span: nil, name: "langchain.llm.call", parent: nil},
        %{span: nil, name: "langchain.chain.execute", parent: nil}
      ])

      orphans = HierarchyValidator.orphaned_spans()
      assert length(orphans) == 2
    end
  end

  describe "validate/0" do
    test "returns empty list for clean stack" do
      assert HierarchyValidator.validate() == []
    end

    test "returns issues for non-empty stack (orphaned spans)" do
      Process.put(:langchain_otel_span_stack, [
        %{span: nil, name: "langchain.llm.call", parent: nil}
      ])

      issues = HierarchyValidator.validate()
      assert length(issues) >= 1
      assert Enum.any?(issues, &(&1.type == :missing_close))
    end

    test "detects invalid parent-child relationships" do
      # Simulate message.process as child of llm.call (invalid — message.process
      # should be under chain.execute)
      Process.put(:langchain_otel_span_stack, [
        %{span: nil, name: "langchain.message.process", parent: nil},
        %{span: nil, name: "langchain.llm.call", parent: nil},
        %{span: nil, name: "langchain.chain.execute", parent: nil}
      ])

      issues = HierarchyValidator.validate()
      # Should detect the invalid nesting
      assert length(issues) >= 1
      assert Enum.any?(issues, &(&1.type in [:invalid_parent, :missing_close]))
    end
  end

  describe "strict mode" do
    test "enable_strict!/0 and disable_strict!/0 toggle the flag" do
      refute HierarchyValidator.strict?()
      HierarchyValidator.enable_strict!()
      assert HierarchyValidator.strict?()
      HierarchyValidator.disable_strict!()
      refute HierarchyValidator.strict?()
    end
  end
end
