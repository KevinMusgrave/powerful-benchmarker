import argparse
import glob
import os


def main(args):
    matches = glob.glob(os.path.join(args.input_folder, "**", "*.tex"))
    for m in matches:
        print(m)
        with open(m, "r") as f:
            newText = f.read().replace(
                "DCP & DCR & DCS & DPC & DPR & DPS & DRC & DRP & DRS & DSC & DSP & DSR",
                "CP & CR & CS & PC & PR & PS & RC & RP & RS & SC & SP & SR",
            )
        with open(m, "w") as f:
            f.write(newText)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input_folder", type=str, default="tables_latex")
    args = parser.parse_args()
    main(args)
