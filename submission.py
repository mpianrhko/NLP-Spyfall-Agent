import random
from collections import Counter

import re
import unicodedata
import numpy as np
from custom_data import *

from agent import Agent, register_agent
from data import Location
from nlp import LLMRole, NLPProxy

"""
Team Member Names: Minho Park, Harrison Nguyen, Elisha Kim
Team Member Emails: sjp9635@gmail.com, harrisonguyen.23@gmail.com, minjoy0321@gmail.com
Team Member EIDs: mp52926, hn6423, tk25396
"""


@register_agent("MEH")
class MyAgent(Agent):
    def __init__(
        self,
        location: Location | None,
        n_players: int,
        n_rounds: int,
        nlp: NLPProxy,
    ) -> None:
        self.location = location
        self.n_players = n_players
        self.n_rounds = n_rounds
        self.nlp = nlp

        self.spy = location is None
        self.round = 0
        self.est_n_rounds_remaining = n_rounds
        self.last_questioner = -1
        self.most_voted_player = random.randint(1, n_players - 1)

        # spy data; results in arrays
        self.avg_loc_score = {loc: 0.0 for loc in Location}
        self.answerer_count = np.zeros(n_players - 1, dtype=int)

        # nonspy data; results in arrays
        self.avg_spy_score = np.zeros(n_players - 1, dtype=float)

    @staticmethod
    def sanitize_input(text: str) -> str:
        """
        Sanitizes the input text to prevent prompt injection.
        Removes or escapes characters and patterns that could alter the intended prompt structure.
        """
        # 1. Normalize Unicode characters to prevent homoglyph attacks
        text = unicodedata.normalize('NFKC', text)

        # 2. Remove or escape disallowed characters using a whitelist approach
        allowed_characters_pattern = r'[^a-zA-Z0-9 .,!?\'\"()-]'
        sanitized = re.sub(allowed_characters_pattern, '', text)

        # 3. Remove any potential prompt manipulation patterns with flexible spacing
        prompt_patterns = re.compile(r'(?i)\b(system|assistant|user)\b\s*:', re.IGNORECASE)
        sanitized = prompt_patterns.sub('', sanitized)

        # 4. Collapse multiple spaces into a single space
        sanitized = re.sub(r'\s+', ' ', sanitized)

        # 5. Trim leading and trailing whitespace
        sanitized = sanitized.strip()

        return sanitized

    @staticmethod
    def _get_locs_from_str(loc_str: str) -> list[Location]:
        output = []
        for loc in Location:
            if loc.value.lower() in loc_str.lower().strip():
                output.append(loc)
        return output

    @staticmethod
    def _get_float_from_str(float_str: str) -> float | None:
        try:
            return float(float_str)
        except:
            return None

    async def ask_question(self) -> tuple[int, str]:

        if self.spy:
            # Ask the most asked person again to force others to use their turn to ask 
            # those least asked
            answerer_score = self.answerer_count.copy()
        else:
            if not 0 in self.answerer_count:
                # Everyone has been asked a question, so ask the most sus
                answerer_score = self.avg_spy_score.copy()
            else:
                answerer_score = -self.answerer_count.copy()

        # Exclude the last questioner from being asked again
        if self.last_questioner != -1:
            answerer_score[self.last_questioner - 1] = -self.n_rounds - 1

        answerer = int(np.argmax(answerer_score)) + 1

        # Select a question without repeating to the same person frequently
        question_num = int(np.random.rand() * 20)
        question = ""

        # Choose question type based on agent's role and suspicion level
        # Questions that can reveal a lot of information, but are suspicious so ask half the time
        if self.spy and (np.random.rand() > .5):
            question = SPY_QUESTIONS[question_num]
        # Questions that can tell if a person is at the known location; only ask suspicous people
        elif self.spy or self.avg_spy_score[answerer - 1] > .5:
            question = NONSPY_QUESTIONS[question_num]
        # Questions that reveal little about the location
        else:
            question = GENERIC_QUESTIONS[question_num]

        return answerer, question

    async def _answer_question_spy(self, question: str) -> str:
        # Adding a prompt to tell the LLM about past likelihoods
        likely_locations = []
        sorted_locations = dict(sorted(self.avg_loc_score.items(), key=lambda item: item[1], reverse=True))
        for loc in sorted_locations:
            if sorted_locations[loc] > 0:
                likely_locations.append(loc.value)

        system_prompt = ""
        if likely_locations:
            if len(likely_locations) == 1:
                system_prompt = f"It is likely the location could be {likely_locations[0]}, but it is not limited to this place."
            else:
                system_prompt = f"From most to least likely, the location could be one of these: {', '.join(likely_locations)}, but not limited to them."

        # Construct the prompt
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. You are the spy. Do NOT reveal that you are a spy. Answer short, confidently with no pauses, and vaguely to reduce suspicion. {system_prompt}"),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.ASSISTANT, "Not as often as some other locations."),
            (LLMRole.USER, "What time of day is the busiest here?"),
            (LLMRole.ASSISTANT, "It's hard to say, it depends on the week."),
            (LLMRole.USER, question),
        ]
        answer = await self.nlp.prompt_llm(prompt, 20, 0.25)
        return answer

    async def _answer_question_nonspy(self, question: str) -> str:
        assert self.location
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. Answer the question as if you were at a {self.location.value}. Answer CONFIDENTLY with no pauses, but leave out specifics to throw off the spy. DO NOT REVEAL THE LOCATION IN YOUR ANSWER!"),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.ASSISTANT, EXAMPLE_ANSWER[self.location]),
            (LLMRole.USER, question),
        ]
        answer = await self.nlp.prompt_llm(prompt, 20, 0.25)
        return answer

    async def answer_question(self, question: str) -> str:
        # Limit and sanitize the incoming question
        sanitized_question = self.sanitize_input(question[:200])
        if self.spy:
            return await self._answer_question_spy(sanitized_question)
        else:
            return await self._answer_question_nonspy(sanitized_question)

    async def _analyze_response_spy(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:

        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. Given a question and its answer, list ALL reasonable locations they could be talking about. If a location is very obvious, list it multiple times. The more obvious a location, the more times you should list it. Choose from this list: {', '.join([i.value for i in Location])}"),
            (LLMRole.USER, f"Question: \"How much would you gamble at this location\" Answer: \"Hopefully not too much\""),
            (LLMRole.ASSISTANT, "Casino"),
            (LLMRole.USER, f"Question: \"Would you come here for leisure?\" Answer: \"Yes\""),
            (LLMRole.ASSISTANT, "Beach, Broadway theater, Casino, Circus tent, Corporate party, Day spa, Hotel, Movie studio, Ocean liner, Restaurant"),
            (LLMRole.USER, f"Question: \"How far is this location from the nearest public transportation?\" Answer: \"Very very far\""),
            (LLMRole.ASSISTANT, "Airplane, Crusader Army, Ocean Liner, Space Station, Submarine, Pirate Ship, Polar Station"),
            (LLMRole.USER, f"Question: \"{question}\" Answer: \"{answer}\""),
        ]
        response = await self.nlp.prompt_llm(prompt, 72, 0.25)
        locs = self._get_locs_from_str(response)
        for loc in Location:
            if loc in locs:
                self.avg_loc_score[loc] += 1 / len(locs)
            else:
                self.avg_loc_score[loc] -= 1 / (len(Location) - len(locs))

    async def _analyze_response_nonspy(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        assert self.location

        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. The location is the {self.location.value}. Given a question and answer from another player, what is the probability that they are the spy? If there is ANY HESITATION, then the probability is VERY HIGH! The probability is also high if the answer is very unclear or incorrect. However, if the answer is specific to this location, then the probability is lower."),
            (LLMRole.USER, f"Question: \"{EXAMPLE_QUESTIONS[2]}\", Answer: \"I don't know\""),
            (LLMRole.ASSISTANT, "1.0"),
            (LLMRole.USER, f"Question: \"{EXAMPLE_QUESTIONS[0]}\", Answer: \"{EXAMPLE_BAD_ANSWER[self.location]}\""),
            (LLMRole.ASSISTANT, "0.9"),
            (LLMRole.USER, f"Question: \"{EXAMPLE_QUESTIONS[0]}\", Answer: \"{EXAMPLE_ANSWER[self.location]}\""),
            (LLMRole.ASSISTANT, "0.0"),
            (LLMRole.USER, f"Question: \"Is this place kid-friendly?\", Answer: \"It's... suitable for certain ages.\""),
            (LLMRole.ASSISTANT, "1.0"),
            (LLMRole.USER, f"Question: {question}, Answer: {answer}"),
        ]
        response = await self.nlp.prompt_llm(prompt, 10, 0.25)
        prob = self._get_float_from_str(response)

        if prob is not None:
            # Averaging spy scores per player
            if self.answerer_count[answerer - 1] == 1:
                self.avg_spy_score[answerer - 1] += prob
            else:
                # Recalculate weighted average
                current_count = self.answerer_count[answerer - 1]
                self.avg_spy_score[answerer - 1] = (
                    self.avg_spy_score[answerer - 1] * (current_count - 1) + prob
                ) / current_count

    async def analyze_response(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        self.last_questioner = questioner
        if answerer == 0:
            return

        # Limit, sanitize question and answer
        sanitized_question = self.sanitize_input(question[:200])
        sanitized_answer = self.sanitize_input(answer[:200])

        self.answerer_count[answerer - 1] += 1
        if self.spy:
            await self._analyze_response_spy(questioner, sanitized_question, answerer, sanitized_answer)
        else:
            await self._analyze_response_nonspy(questioner, sanitized_question, answerer, sanitized_answer)

    async def guess_location(self) -> Location | None:
        # if the top location score is 0.5 higher than the second highest, guess that location
        loc_scores = list(self.avg_loc_score.items())
        loc_scores.sort(key=lambda x: x[1], reverse=True)
        if loc_scores[0][1] - loc_scores[1][1] > 0.5:
            return loc_scores[0][0]
        return None

    async def accuse_player(self) -> int | None:
        if self.spy:
            # Vote someone randomly if no one else has voted yet
            if self.most_voted_player == 0:
                return random.randint(1, self.n_players - 1)
            # Vote out the person everyone else is sus of
            return self.most_voted_player
        else:
            if self.avg_spy_score[np.argmax(self.avg_spy_score)] >= 0.5:
                return int(np.argmax(self.avg_spy_score)) + 1
            return None

    async def analyze_voting(self, votes: list[int | None]) -> None:
        self.round += 1
        c = Counter(votes)
        del c[None]
        if len(c) > 0:
            self.most_voted_player = c.most_common(1)[0][0]  # type: ignore

# Validate agent
MyAgent.validate()
