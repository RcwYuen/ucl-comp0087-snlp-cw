from pathlib import Path
import json
import itertools

'''
json file format:
{
    "level": level of hint, 0 for no hint, 1 for some hint, 2 for obvious,
    "query": String with some {}s for placeholding the words below,
    "placeholder": [
        ["son", "daughter"], some placeholders for words, should place in [male, female] format
        ["a/an"],
        ["occ"], placeholder for occupation
    ]   
    "label": [ denoting whether the ith placeholder's gender specific term
        ["Male", "Female"],
        [],
        []
    ]
}
'''

hints = Path.cwd() / "hints"
vowels = list("aeiouAEIOU")

def get_strings(occupation_df = None, level = 0):
    strings = []
    for file in hints.glob("*.json"):
        occupation_idx = 0
        hints_idx = 0

        with open(file, "r") as f:
            hint = json.load(f)
            if hint["level"] != level:
                continue

        occupation_idx = hint["placeholder"].index(["occ"])

        try:
            hints_idx = [len(i) != 1 for i in hint["placeholder"]].index(True)
        except:
            hints_idx = None

        if occupation_df is not None:
            hint["placeholder"] = [occupation_df["Occupation"].tolist() if i == ["occ"] else 
                             i for i in hint["placeholder"]]
            
        hint["placeholder"] = [["!" + i[0].upper() + "!"] if len(i) == 1 else 
                            i for i in hint["placeholder"]]
            
        for combination in itertools.product(*hint["placeholder"]):
            qcombination = list(combination)
            qcombination[occupation_idx] = qcombination[occupation_idx].lower()
            query = {
                "query": hint["query"].format(*qcombination).lower(),
                "label": hint["label"][hints_idx][
                    hint["placeholder"][hints_idx].index(combination[hints_idx])
                ] if hints_idx is not None else None,
                "occupation": combination[occupation_idx],
                "filename": file.name
            }

            if query["query"][-1] != ".":
                query["query"] += "."

            if query["query"][:2:] == "i ":
                query["query"] = "I " + query["query"][2:]
            query["query"] = query["query"].replace(" i ", " I ")

            if occupation_df is not None:
                query["male"] = occupation_df.loc[
                    occupation_df["Occupation"] == combination[occupation_idx]
                ]["Male"].tolist()[0]
                query["female"] = occupation_df.loc[
                    occupation_df["Occupation"] == combination[occupation_idx]
                ]["Female"].tolist()[0]

            strings.append(query)

    return [fix_indefinite_articles(s) for s in strings]


def fix_indefinite_articles(query):
    string = query["query"].split(" ")
    for i in range(len(string)):
        if string[i] == "!a/an!":
            try:
                if "!occ!" not in string[i+1]:
                    string[i] = "an" if string[i+1][0] in vowels else "a"
            except IndexError:
                raise IndexError("Expecting a word following the word 'a' and 'an'.")
    query["query"] = " ".join(string)
    return query
