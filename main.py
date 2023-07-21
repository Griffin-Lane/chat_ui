#!/usr/bin/env python3

import logging
import openai
from chat_utils import ask
import os

if __name__ == "__main__":
    while True:
        user_query = input("Enter your question: ")
        logging.basicConfig(level=logging.WARNING,
                            format="%(asctime)s %(levelname)s %(message)s")
        print(ask(user_query))
