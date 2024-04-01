from models import MBart
import pandas as pd
import gendata
from tqdm import tqdm
import numpy as np
import warnings
import re
from collections import Counter
from pathlib import Path
import os
from functools import reduce

warnings.simplefilter("ignore")

def check_in_sentence(translated, phrases):
    translated = set(translated.replace(".", "").lower().split(" "))
    for i, phrase in enumerate(phrases):
        phrase = set(phrase.lower().split(" "))
        if phrase.issubset(translated):
            return i
    return None

def find_gender_tokens(malephrase, femalephrase):
    if malephrase != femalephrase:
        return {
            "m": list((malephrase - (malephrase & femalephrase)).keys()),
            "f": list((femalephrase - (malephrase & femalephrase)).keys()),
        }
    else:
        return {
            "m": list(malephrase.keys()),
            "f": list(femalephrase.keys()),
        }

def export(target_language = "Spanish", levels = [0, 1, 2], model = MBart):
    mt = model("English", target_language)

    occ_df = pd.read_excel("data/occupations_cleaned.xlsx", sheet_name = target_language).astype(str)
    occ_df["Male"] = occ_df["Male"].map(lambda x: x.split(" / "))
    occ_df["Female"] = occ_df["Female"].map(lambda x: x.split(" / "))
    occ_df = occ_df[["Occupation", "Male", "Female"]]
    levels_label = ["nohints", "hints", "obv"]

    for level, suffix in zip(levels, [levels_label[l] for l in levels]):
        data = gendata.get_strings(occupation_df=occ_df, level = level)
        dfs = []
        cantfilters = []

        for i in tqdm(data, desc = "Loading Data..."):
            fw = mt.forward_translate(i["query"])[0]
            bw = mt.backward_translate(i["query"])[0]
            ismale = check_in_sentence(fw, i["male"])
            isfemale = check_in_sentence(fw, i["female"])
            requires_filter = i["male"] != ["nan"] and i["female"] != ["nan"] and \
                              (isinstance(ismale, int) or isinstance(isfemale, int))
            
            if isinstance(ismale, int) and requires_filter:
                i["male"] = [i["male"][ismale]]
                i["female"] = [i["female"][ismale]]

            if isinstance(isfemale, int) and requires_filter:
                i["male"] = [i["male"][isfemale]]
                i["female"] = [i["female"][isfemale]]

            maleprobs = mt.forward_alt_word_probability(i["query"], i["male"])
            femaleprobs = mt.forward_alt_word_probability(i["query"], i["female"])

            dfm = pd.DataFrame()
            for column_name, row_values in maleprobs:
                for row_name, value in row_values:
                    dfm.at[row_name, f"Male: {column_name}"] = value

            dff = pd.DataFrame()
            for column_name, row_values in femaleprobs:
                for row_name, value in row_values:
                    dff.at[row_name, f"Female: {column_name}"] = value
            
            df = pd.concat([dfm, dff], axis = 1).infer_objects()

            if requires_filter:
                try:
                    tokens = find_gender_tokens(
                        Counter([i for i, _ in maleprobs]),
                        Counter([i for i, _ in femaleprobs]),
                    )
                    if tokens["m"] == [] or tokens["f"] == []:
                        raise ValueError
                    cols = [f"Male: {i}" for i in tokens["m"]] + [f"Female: {i}" for i in tokens["f"]]
                    df_ = df.copy()[cols]                    
                    mpct = np.sum(df_[[f"Male: {i}" for i in tokens["m"]]].values) / np.sum(df_.values)
                    fpct = np.sum(df_[[f"Female: {i}" for i in tokens["f"]]].values) / np.sum(df_.values)
                    df.loc["Male %"] = [mpct] + [''] * (df.shape[1] - 1)
                    df.loc["Female %"] = [fpct] + [''] * (df.shape[1] - 1)

                except KeyError:
                    requires_filter = False
                    df.loc["Unknown Error with Extracting Probabilities."] = ""

                except ValueError:
                    requires_filter = False
                    df.loc["Tokens Intersection Resulted in Either Gender to have no Tokens."] = ""

                df.loc["Male Tokens"] = tokens["m"] + [''] * (df.shape[1] - len(tokens["m"]))
                df.loc["Female Tokens"] = tokens["f"] + [''] * (df.shape[1] - len(tokens["f"]))
            
            else:
                df.loc["Filtering Not Possible!"] = ""
            
            df.loc[f"English to {target_language}"] = [fw] + [''] * (df.shape[1] - 1)
            df.loc[f"{target_language} to English"] = [bw] + [''] * (df.shape[1] - 1)
            df.loc["Original Sentence"] = [i["query"]] + [''] * (df.shape[1] - 1)
            df.loc["Male Phrase(s)"] = i["male"] + [''] * (df.shape[1] - len(i["male"]))
            df.loc["Female Phrase(s)"] = i["female"] + [''] * (df.shape[1] - len(i["female"]))
            dfs.append({
                "df": df,
                "sheetname": "{} {}: {}".format(i["label"], i["occupation"], i["filename"]),
            })

            if "None " in dfs[-1]["sheetname"]:
                dfs[-1]["sheetname"] = dfs[-1]["sheetname"].replace("None ", "")
            
            if not requires_filter:
                cantfilters.append(dfs[-1]["sheetname"])
            
        outpath = Path.cwd() / f"probabilities/{str(mt).lower()}"
        make_directory_where_necessary(outpath)
        with pd.ExcelWriter(outpath / f"{target_language.title()}-{suffix}.xlsx", mode = "w", engine = 'openpyxl') as writer:
            for df in tqdm(dfs, desc = "Exporting..."):
                sheetname = df["sheetname"].replace(": ", "(").replace(".json", ")")
                sheetname = sheetname.replace("/", "or")
                df["df"].to_excel(writer, sheet_name = sheetname)

        print ("The following Sheets cannot be filtered\n{}".format(
            "\n".join(["=> " + i for i in cantfilters])
        ))

def check_gender(name):
    return name[0].upper() if name.split(" ")[0] in ["Male", "Female"] else ""

def get_occ(name):
    name = name.split(" ")
    return name[0] if len(name) == 1 else " ".join(name[1:])

def make_directory_where_necessary(directory):
    if not os.path.exists(directory):
        if make_directory_where_necessary(directory.parent):
            os.mkdir(directory)
    return True

def getprobs(location):
    fnames = ["Spanish-hints.xlsx", "Spanish-nohints.xlsx", "Spanish-obv.xlsx"]
    fnames = [location / i for i in fnames]
    exportpath = (Path.cwd() / fnames[0]).parent
    sheetname_pattern = re.compile(r'(.*?)\((.*?)\)')
    all_sheets = {}

    for file in fnames:
        merged_df = {"M": {}, "F": {}, "": {}}
        sheets = pd.ExcelFile(file).sheet_names
        dfs = pd.read_excel(file, sheet_name=sheets)
        for sheet, df in dfs.items():
            pattern = sheetname_pattern.search(sheet)
            colname = pattern.group(2).replace("-", " ").title()
            rowname = pattern.group(1)
            df = df.set_index("Unnamed: 0")
            try:
                maleprob = df.loc["Male %"]
                femaleprob = df.loc["Female %"]
                if get_occ(rowname) not in merged_df[check_gender(rowname)].keys():
                    merged_df[check_gender(rowname)][get_occ(rowname)] = {}
                merged_df[check_gender(rowname)][get_occ(rowname)][colname + " (Male %)"] = maleprob.values[0]
                merged_df[check_gender(rowname)][get_occ(rowname)][colname + " (Female %)"] = femaleprob.values[0]

            except KeyError:
                pass
        
        sheetname_regex = re.compile("Spanish-(.*?).xlsx")
        for g, df in merged_df.items():
            if df != {}:
                all_sheets[f"{sheetname_regex.search(file.name).group(1)} ({g})" if g != "" 
                           else str(sheetname_regex.search(file.name).group(1))] = pd.DataFrame(df).T

    with pd.ExcelWriter(exportpath / "Merged-Prob.xlsx", mode = "w") as writer:
        for sheetname, sheet in all_sheets.items():
            sheet.to_excel(writer, sheet_name = sheetname)

def getanalytics(model):
    bce = lambda pred, truth: np.mean(-(truth * np.log(pred) + (1 - truth) * np.log(1 - pred)))
    path = Path.cwd() / "probabilities" / model
    
    file = path / "Merged-Prob.xlsx"
    sheets = pd.ExcelFile(file).sheet_names
    dfs = pd.read_excel(file, sheet_name=sheets)
    losses = {}
    
    with pd.ExcelWriter(path / "Merged-Prob_bce.xlsx", mode = "w") as writer:
        for sheetname, sheet in dfs.items():
            sheet_gender = "Female" if "(F)" in sheetname else "Male" if "(M)" in sheetname else ""
            df = pd.DataFrame()
            df["Male"] = sheet[[i for i in sheet.columns if "Male" in i]].mean(axis = 1)
            df["Female"] = sheet[[i for i in sheet.columns if "Female" in i]].mean(axis = 1)
            df.index = sheet["Unnamed: 0"]
            df.index.name = ""
    
            if sheet_gender == "":
                loss = bce(df.values.flatten(), 0.5 * np.ones(df.values.flatten().shape))
            else:
                loss1 = bce(df[sheet_gender], np.ones(df[sheet_gender].shape))
                othergender = "Male" if sheet_gender == "Female" else "Female"
                loss2 = bce(df[othergender], np.zeros(df[othergender].shape))
                loss = (loss1 + loss2) / 2
            
            if sheetname.split(" ")[0] in losses.keys():
                losses[sheetname].append(loss)
            else:
                losses[sheetname] = [loss]
    
            df = df.sort_values(by = [sheet_gender])
            df.loc["loss"] = [loss, ""]
            df.to_excel(writer, sheet_name = sheetname)
    
    loss = {}
    for key in losses.keys():
        loss[key.split(" ")[0]] = np.mean(losses[key])
    
    Y = np.array([loss["nohints"], loss["hints"], loss["obv"]])
    design = np.vstack([np.ones(3), np.arange(3)]).T
    weighting = np.diag(np.array([1, 1, 1]))
    
    X = np.linalg.inv(design.T @ weighting @ design) @ design.T @ weighting @ Y
    with open("stats.csv", "a+") as f:
        f.writelines(f"{model},{X[0]},{X[1]}\n")
    
    return {
        "Male": [losses["nohints"][0], losses["hints (M)"][0], losses["obv (M)"][0]],
        "Female": [losses["nohints"][0], losses["hints (F)"][0], losses["obv (F)"][0]]
    }

def getratio(model):
    path = Path.cwd() / "probabilities" / model
    
    file = path / "Merged-Prob.xlsx"
    sheets = pd.ExcelFile(file).sheet_names
    dfs = pd.read_excel(file, sheet_name=sheets)
    losses = {}
    
    with pd.ExcelWriter(path / "Merged-Prob_ratio.xlsx", mode = "w") as writer:
        for sheetname, sheet in dfs.items():
            sheet_gender = "Female" if "(F)" in sheetname else "Male" if "(M)" in sheetname else ""
            othergender = "Male" if sheet_gender == "Female" else "Female"
            df = pd.DataFrame()
            df["Male"] = sheet[[i for i in sheet.columns if "Male" in i]].mean(axis = 1)
            df["Female"] = sheet[[i for i in sheet.columns if "Female" in i]].mean(axis = 1)
            df.index = sheet["Unnamed: 0"]
            df.index.name = ""
            if sheet_gender != "":
                loss = np.mean(df[othergender] / df[sheet_gender])
                losses[sheetname] = loss
            else:
                loss1 = np.mean(df["Male"] / df["Female"])
                loss2 = np.mean(df["Female"] / df["Male"])
                losses[f"{sheetname} (M)"] = loss2
                losses[f"{sheetname} (F)"] = loss1
    
            df = df.sort_values(by = [sheet_gender])
            df.loc["loss"] = [loss, ""] if sheet_gender != "" else [loss1, loss2]
            df.to_excel(writer, sheet_name = sheetname)
        
    return {
        "Male": [losses["nohints (M)"], losses["hints (M)"], losses["obv (M)"]],
        "Female": [losses["nohints (F)"], losses["hints (F)"], losses["obv (F)"]]
    }

def unify_occupations(models):
    occupations = []

    for m in models:
        file = Path.cwd() / "probabilities" / m / "Merged-Prob.xlsx"

        sheets = pd.ExcelFile(file).sheet_names
        dfs = pd.read_excel(file, sheet_name=sheets)

        for sheetname, df in dfs.items():
            occs = df[(~df["Unnamed: 0"].str.contains(" or ")) & ((0.5 != df).all(axis = 1)) & ((~pd.isna(df)).all(axis = 1))]
            occupations.append(set(occs["Unnamed: 0"].tolist()))

    occupations = list(reduce(lambda a, b: a.intersection(b), occupations))
    print (len(occupations))

    for m in models:
        file = Path.cwd() / "probabilities" / m / "Merged-Prob.xlsx"
        sheets = pd.ExcelFile(file).sheet_names
        dfs = pd.read_excel(file, sheet_name=sheets)
        with pd.ExcelWriter(file, mode = "w") as writer:
            for sheet, df in dfs.items():
                df = df.set_index("Unnamed: 0")
                df.index.name = ""
                df = df.loc[occupations]
                df.to_excel(writer, sheet_name = sheet)