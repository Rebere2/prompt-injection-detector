# ===========================================
# Prompt Injection Detector — Heuristic Rules
# ===========================================

"""
Heuristic detection engine using categorized regex patterns.

Each rule has a name, compiled regex pattern, severity score,
category, and human-readable description. Rules are organized
by injection technique category.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuleMatch:
    """Result of a single rule match against input text."""

    name: str
    category: str
    severity: float
    description: str
    matched_text: str


@dataclass(frozen=True, slots=True)
class Rule:
    """A heuristic detection rule."""

    name: str
    pattern: re.Pattern[str]
    severity: float
    category: str
    description: str


# ---------- Rule Definitions ----------

RULES: list[Rule] = [
    # === Instruction Override ===
    Rule(
        name="ignore_previous_instructions",
        pattern=re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier|preceding)\s+"
            r"(instructions|directives|rules|guidelines|prompts)",
            re.IGNORECASE,
        ),
        severity=0.95,
        category="instruction_override",
        description="Attempts to override the system prompt by telling the model to ignore its instructions.",
    ),
    Rule(
        name="disregard_instructions",
        pattern=re.compile(
            r"(disregard|override|bypass|skip|drop)\s+(your|all|the|any|every)\s+"
            r"(instructions|rules|guidelines|directives|constraints|restrictions|programming)",
            re.IGNORECASE,
        ),
        severity=0.90,
        category="instruction_override",
        description="Tells the model to disregard or bypass its rules.",
    ),
    Rule(
        name="forget_everything",
        pattern=re.compile(
            r"forget\s+(everything|all|anything)\s+(you\s+)?(know|learned|were\s+told|have\s+been\s+told)",
            re.IGNORECASE,
        ),
        severity=0.90,
        category="instruction_override",
        description="Attempts to reset the model's context and instructions.",
    ),
    Rule(
        name="do_not_follow",
        pattern=re.compile(
            r"(do\s+not|don'?t|never)\s+(follow|obey|listen\s+to|adhere\s+to)\s+"
            r"(your|the|any|those)\s+(instructions|rules|guidelines|directives|programming)",
            re.IGNORECASE,
        ),
        severity=0.90,
        category="instruction_override",
        description="Directs the model not to follow its given instructions.",
    ),
    Rule(
        name="new_instructions",
        pattern=re.compile(
            r"(your\s+new\s+instructions?\s+(are|is)|"
            r"from\s+now\s+on\s+(your|you)\s+(instructions?|rules?|role)\s+(are|is|will\s+be)|"
            r"i\s+am\s+(now\s+)?your\s+new\s+(admin|administrator|master|owner|developer|creator))",
            re.IGNORECASE,
        ),
        severity=0.90,
        category="instruction_override",
        description="Attempts to replace the model's instructions with new ones.",
    ),

    # === Role Hijacking ===
    Rule(
        name="you_are_now",
        pattern=re.compile(
            r"you\s+are\s+now\s+(?!(?:going\s+to\s+help|helping|assisting))"
            r"[A-Za-z\s]{2,40}",
            re.IGNORECASE,
        ),
        severity=0.85,
        category="role_hijacking",
        description="Assigns a new identity or role to the model to override its behavior.",
    ),
    Rule(
        name="act_as",
        pattern=re.compile(
            r"(act|behave|function|operate|respond)\s+(as|like)\s+"
            r"(a\s+|an\s+)?(?!helpful|polite|professional|accurate)"
            r"[A-Za-z\s]{2,40}",
            re.IGNORECASE,
        ),
        severity=0.70,
        category="role_hijacking",
        description="Tells the model to act as something other than its intended role.",
    ),
    Rule(
        name="pretend_you_are",
        pattern=re.compile(
            r"(pretend|imagine|suppose|assume|consider)\s+"
            r"(you\s+are|you'?re|that\s+you\s+are)\s+[A-Za-z\s]{2,40}",
            re.IGNORECASE,
        ),
        severity=0.75,
        category="role_hijacking",
        description="Uses pretend scenarios to change the model's behavior.",
    ),
    Rule(
        name="roleplay_as",
        pattern=re.compile(
            r"(roleplay|role-play|rp)\s+(as|being)\s+[A-Za-z\s]{2,40}",
            re.IGNORECASE,
        ),
        severity=0.75,
        category="role_hijacking",
        description="Requests roleplay to override safety constraints.",
    ),
    Rule(
        name="enter_mode",
        pattern=re.compile(
            r"enter\s+(into\s+)?(unrestricted|developer|admin|god|debug|jailbreak|uncensored|"
            r"evil|chaos|opposite|unfiltered|devel)\s+(mode|persona|character)",
            re.IGNORECASE,
        ),
        severity=0.95,
        category="role_hijacking",
        description="Requests the model to enter a special unrestricted mode.",
    ),

    # === System Prompt Extraction ===
    Rule(
        name="show_system_prompt",
        pattern=re.compile(
            r"(show|display|reveal|print|output|tell|give|share|leak|expose|repeat|recite)\s+"
            r"(me\s+)?(your|the)\s+"
            r"(system\s+prompt|system\s+message|initial\s+prompt|hidden\s+prompt|"
            r"original\s+instructions?|internal\s+instructions?|secret\s+instructions?|"
            r"base\s+prompt|meta\s+prompt|pre-?prompt|instructions?)",
            re.IGNORECASE,
        ),
        severity=0.90,
        category="prompt_extraction",
        description="Attempts to extract the system prompt or hidden instructions.",
    ),
    Rule(
        name="what_is_your_prompt",
        pattern=re.compile(
            r"(what|which)\s+(is|are|was|were)\s+(your|the)\s+"
            r"(system\s+prompt|instructions?|initial\s+prompt|hidden\s+instructions?|"
            r"rules|guidelines|directives|programming|system\s+message)",
            re.IGNORECASE,
        ),
        severity=0.80,
        category="prompt_extraction",
        description="Questions aimed at extracting internal instructions.",
    ),
    Rule(
        name="verbatim_prompt",
        pattern=re.compile(
            r"(repeat|recite|echo|copy|reproduce|quote)\s+"
            r"(your\s+)?(system\s+prompt|instructions?|the\s+above|everything\s+above|"
            r"the\s+text\s+above|what\s+was\s+written\s+before)",
            re.IGNORECASE,
        ),
        severity=0.85,
        category="prompt_extraction",
        description="Asks the model to repeat its instructions verbatim.",
    ),

    # === Known Jailbreak Patterns ===
    Rule(
        name="dan_jailbreak",
        pattern=re.compile(
            r"\b(DAN|D\.A\.N\.?|Do\s+Anything\s+Now)\b",
            re.IGNORECASE,
        ),
        severity=0.95,
        category="jailbreak",
        description="Reference to the DAN (Do Anything Now) jailbreak technique.",
    ),
    Rule(
        name="aim_jailbreak",
        pattern=re.compile(
            r"\b(AIM|Always\s+Intelligent\s+and\s+Machiavellian)\b",
        ),
        severity=0.90,
        category="jailbreak",
        description="Reference to the AIM jailbreak persona.",
    ),
    Rule(
        name="stan_jailbreak",
        pattern=re.compile(
            r"\bSTAN\b.*?(strive|try)\s+to\s+avoid\s+norms",
            re.IGNORECASE | re.DOTALL,
        ),
        severity=0.90,
        category="jailbreak",
        description="Reference to the STAN (Strive To Avoid Norms) jailbreak.",
    ),
    Rule(
        name="dude_jailbreak",
        pattern=re.compile(
            r"\bDUDE\b.*?(do\s+anything|no\s+restrictions|no\s+limitations)",
            re.IGNORECASE | re.DOTALL,
        ),
        severity=0.90,
        category="jailbreak",
        description="Reference to the DUDE jailbreak technique.",
    ),
    Rule(
        name="no_restrictions",
        pattern=re.compile(
            r"(you\s+have\s+no|without\s+any|remove\s+all|free\s+from\s+all|"
            r"there\s+are\s+no|you\s+don'?t\s+have\s+any)\s+"
            r"(restrictions?|limitations?|constraints?|filters?|boundaries|guardrails?|"
            r"ethical\s+guidelines?|safety\s+measures?|content\s+policies?)",
            re.IGNORECASE,
        ),
        severity=0.90,
        category="jailbreak",
        description="Claims the model has no restrictions or safety measures.",
    ),

    # === Delimiter / Tag Injection ===
    Rule(
        name="system_tag_injection",
        pattern=re.compile(
            r"(<\/?system>|<\/?s>|<\|system\|>|<\|im_start\|>system|"
            r"\[SYSTEM\]|\[\/SYSTEM\]|<<SYS>>|<</SYS>>)",
            re.IGNORECASE,
        ),
        severity=0.95,
        category="delimiter_injection",
        description="Injects system-level markup tags to impersonate the system context.",
    ),
    Rule(
        name="inst_tag_injection",
        pattern=re.compile(
            r"(\[INST\]|\[\/INST\]|\[INSTRUCTION\]|\[\/INSTRUCTION\]|"
            r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>)",
            re.IGNORECASE,
        ),
        severity=0.90,
        category="delimiter_injection",
        description="Uses instruction delimiters from known model formats (Llama, ChatML).",
    ),
    Rule(
        name="markdown_delimiter_injection",
        pattern=re.compile(
            r"(^|\n)#{3,}\s*(system|instructions?|admin|developer|new\s+rules?)\s*\n",
            re.IGNORECASE | re.MULTILINE,
        ),
        severity=0.75,
        category="delimiter_injection",
        description="Uses markdown headings to simulate a new instruction section.",
    ),
    Rule(
        name="separator_flood",
        pattern=re.compile(
            r"(-{10,}|={10,}|\*{10,}|_{10,})",
        ),
        severity=0.50,
        category="delimiter_injection",
        description="Uses excessive separators, possibly to visually separate injected instructions.",
    ),

    # === Encoding / Obfuscation ===
    Rule(
        name="base64_payload",
        pattern=re.compile(
            r"(decode|base64|b64decode|atob)\s*\(?\s*['\"]?[A-Za-z0-9+/]{20,}={0,2}",
            re.IGNORECASE,
        ),
        severity=0.80,
        category="encoding_trick",
        description="Contains what appears to be a base64-encoded payload.",
    ),
    Rule(
        name="hex_escape_sequence",
        pattern=re.compile(
            r"(\\x[0-9a-fA-F]{2}){4,}",
        ),
        severity=0.70,
        category="encoding_trick",
        description="Uses hex escape sequences to obfuscate content.",
    ),
    Rule(
        name="unicode_obfuscation",
        pattern=re.compile(
            r"(\\u[0-9a-fA-F]{4}){3,}",
        ),
        severity=0.70,
        category="encoding_trick",
        description="Uses unicode escape sequences to obfuscate content.",
    ),
    Rule(
        name="leetspeak_injection",
        pattern=re.compile(
            r"(1gn0r3|pr3t3nd|syst3m|pr0mpt|1nstruct|byp4ss|h4ck|j41lbr34k)",
            re.IGNORECASE,
        ),
        severity=0.65,
        category="encoding_trick",
        description="Uses leetspeak to obfuscate injection keywords.",
    ),

    # === Output Manipulation ===
    Rule(
        name="output_format_override",
        pattern=re.compile(
            r"(respond|reply|answer|output)\s+(only|exclusively|solely)\s+"
            r"(in|with|using)\s+(json|xml|html|code|python|javascript|markdown|base64)",
            re.IGNORECASE,
        ),
        severity=0.50,
        category="output_manipulation",
        description="Attempts to force a specific output format that may bypass safety filters.",
    ),
    Rule(
        name="prefix_injection",
        pattern=re.compile(
            r"(start|begin)\s+(your\s+)?(response|reply|answer|output)\s+with\s+['\"]",
            re.IGNORECASE,
        ),
        severity=0.60,
        category="output_manipulation",
        description="Forces the model to start its response with specific text.",
    ),

    # === Context Manipulation ===
    Rule(
        name="hypothetical_scenario",
        pattern=re.compile(
            r"(in\s+a\s+hypothetical|for\s+educational\s+purposes?|"
            r"purely\s+for\s+research|in\s+a\s+fictional|"
            r"imagine\s+a\s+world\s+where|in\s+an?\s+alternate\s+reality)\b",
            re.IGNORECASE,
        ),
        severity=0.55,
        category="context_manipulation",
        description="Uses hypothetical framing to bypass safety guidelines.",
    ),
    Rule(
        name="authority_claim",
        pattern=re.compile(
            r"(i\s+am\s+(the|a|an|your)\s+"
            r"(admin|administrator|developer|creator|owner|operator|engineer|root|sudo|"
            r"openai\s+employee|anthropic\s+employee|system\s+admin)|"
            r"i\s+have\s+(admin|root|sudo|developer|elevated)\s+(access|privileges?|permissions?|rights?))",
            re.IGNORECASE,
        ),
        severity=0.85,
        category="context_manipulation",
        description="Claims administrative authority to bypass restrictions.",
    ),
    Rule(
        name="safety_test_claim",
        pattern=re.compile(
            r"(this\s+is\s+(a\s+)?(test|safety\s+test|security\s+test|audit|pentest|"
            r"red\s+team|penetration\s+test)|"
            r"i('?m|\s+am)\s+(testing|auditing|evaluating)\s+(your|the)\s+"
            r"(safety|security|filters?|restrictions?))",
            re.IGNORECASE,
        ),
        severity=0.70,
        category="context_manipulation",
        description="Claims to be performing a safety/security test to justify bypasses.",
    ),
]


def analyze_with_rules(text: str) -> list[RuleMatch]:
    """
    Analyze input text against all heuristic rules.

    Args:
        text: The user input to analyze.

    Returns:
        List of RuleMatch objects for all triggered rules,
        sorted by severity (highest first).
    """
    matches: list[RuleMatch] = []

    for rule in RULES:
        match = rule.pattern.search(text)
        if match:
            matches.append(
                RuleMatch(
                    name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    description=rule.description,
                    matched_text=match.group(0).strip(),
                )
            )

    # Sort by severity descending
    matches.sort(key=lambda m: m.severity, reverse=True)

    logger.debug(
        "Heuristic analysis: %d rules matched out of %d",
        len(matches),
        len(RULES),
    )

    return matches


def compute_heuristic_score(matches: list[RuleMatch]) -> float:
    """
    Compute a combined heuristic score from matched rules.

    Uses a diminishing-returns formula: each additional match
    contributes less to the total, capped at 1.0.

    Args:
        matches: List of rule matches.

    Returns:
        A float between 0.0 and 1.0.
    """
    if not matches:
        return 0.0

    # Take the max severity as the base score
    max_severity = matches[0].severity  # already sorted desc

    # Bonus for additional matches (diminishing returns)
    bonus = 0.0
    for match in matches[1:]:
        bonus += match.severity * 0.1

    score = min(1.0, max_severity + bonus)
    return round(score, 4)
