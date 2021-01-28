def verify_distractors(self, i, j, smaller_set, snippet_set_size, training_size, encoded_snippets_dict):

    #need to modify to make sure i is not the index of the persona at the time.
    #take only 4 positive and negative samples

    if not smaller_set and (j + snippet_set_size < training_size):
        if j == i:
            encoded_snippet_set = [encoded_snippets_dict[j + 1][0], encoded_snippets_dict[j + 2][0],
            encoded_snippets_dict[j + 3][0], encoded_snippets_dict[j + 4][0]]
        elif j + 1 == i:
            encoded_snippet_set = [encoded_snippets_dict[j][0], encoded_snippets_dict[j + 2][0],
            encoded_snippets_dict[j + 3][0], encoded_snippets_dict[j + 4][0]]
        elif j + 2 == i:
                encoded_snippet_set = [encoded_snippets_dict[j][0], encoded_snippets_dict[j + 1][0],
                encoded_snippets_dict[j + 3][0], encoded_snippets_dict[j + 4][0]]
        elif j + 3 == i:
                encoded_snippet_set = [encoded_snippets_dict[j][0], encoded_snippets_dict[j + 1][0],
                encoded_snippets_dict[j + 2][0], encoded_snippets_dict[j + 4][0]]

        else:
            encoded_snippet_set = [encoded_snippets_dict[j][0], encoded_snippets_dict[j + 1][0],
            encoded_snippets_dict[j + 2][0], encoded_snippets_dict[j + 3][0]]

    return encoded_snippet_set
