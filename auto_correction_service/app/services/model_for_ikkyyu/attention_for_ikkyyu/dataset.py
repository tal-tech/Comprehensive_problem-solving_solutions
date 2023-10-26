START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]


def load_vocab_market(tokens_file):
    with open(tokens_file, "rb") as fd:
        tokens = fd.read()
        tokens = tokens.decode('utf-8', errors="surrogateescape")
        tokens = tokens.split('\t')
        tokens.extend(SPECIAL_TOKENS)
        token_to_id = {tok: i for i, tok in enumerate(tokens)}
        id_to_token = {i: tok for i, tok in enumerate(tokens)}
        return token_to_id, id_to_token

