"""
Python script for manually generating bcrypt hashes for Anything LLM users, 
e.g. for recovering access to Anything LLM by replacing hash in user table

not recommended / not standard practice
"""

#!/usr/bin/env python3

import bcrypt
import getpass  # For secure password input (hides input)

# Prompt user for password interactively
password = getpass.getpass("Enter a new password: ")

# Generate a salt with cost factor 10 (matches AnythingLLM's default)
salt = bcrypt.gensalt(rounds=10)

# Hash the password
hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)

# Output the hash
print("Generated bcrypt hash:", hashed_password.decode('utf-8'))