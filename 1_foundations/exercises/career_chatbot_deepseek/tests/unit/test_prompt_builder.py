from utils.prompt_builder import PromptBuilder


def test_build_system_prompt():
    builder = PromptBuilder()
    prompt = builder.build_system_prompt(
        name="Test Agent",
        linkedin_profile="LinkedIn info",
        summary="Summary info"
    )

    assert "Test Agent" in prompt
    assert "LinkedIn info" in prompt
    assert "Summary info" in prompt


def test_build_system_prompt_with_rag():
    builder = PromptBuilder()
    prompt = builder.build_system_prompt(
        name="Test",
        linkedin_profile="",
        summary="",
        enable_rag=True
    )

    assert "search_knowledge_base" in prompt


def test_build_system_prompt_without_rag():
    builder = PromptBuilder()
    prompt = builder.build_system_prompt(
        name="Test",
        linkedin_profile="",
        summary="",
        enable_rag=False
    )

    assert "search_knowledge_base" not in prompt