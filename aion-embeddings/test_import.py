import sys
try:
    import sentence_transformers
    print("SUCCESS: sentence_transformers imported!")
except Exception as e:
    print(f"EXCEPTION: {e}")
except BaseException as e:
    print(f"BASE EXCEPTION: {e}")
