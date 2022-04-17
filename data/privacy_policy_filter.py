import os
import shutil

alexa = "alexa_rank.txt"
dest_folder = "privacy_policy_alexa"
top_n = 500
if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)
with open(alexa) as f:
    blank = f.readline()
    for i in range(top_n):
        website = f.readline()
        category = f.readline()
        blank = f.readline()
        # privacy-policy-historical is cloned from:
        # https://github.com/citp/privacy-policy-historical
        privacy_policy_path = (
            "privacy-policy-historical"
            + "/"
            + website[0]
            + "/"
            + website[0:2]
            + "/"
            + website[0:3]
            + "/"
            + website[:-1]
            + ".md"
        )
        shutil.copy(privacy_policy_path, dest_folder)
