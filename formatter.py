
import re


def format_response(answer):

    answer = answer.strip()

    if not answer:
        return "Sorry, I couldn't generate an answer."

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return answer

    # First sentence as explanation
    explanation = sentences[0]

    # Remaining sentences as key points
    points = sentences[1:]

    formatted_answer = f"### Explanation\n{explanation}"

    if points:
        formatted_answer += "\n\n### Key Points\n"

        for point in points[:3]:
            formatted_answer += f"- {point}\n"

    return formatted_answer
