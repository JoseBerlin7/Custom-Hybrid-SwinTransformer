
def choice_heads(embed_dim, preferred):
    h = max(1, preferred)
    while embed_dim % h !=0:
        h -= 1
    return h