import itertools
import csv

def commands(predict_types: list, top_k = []):
    layers = range(1,9)  # 0–7
    if top_k == []: min_len, max_len = 2, 6
    else: 
        if len(top_k) == 1 and top_k[0] == 0: desired_length = 1
        else: desired_length = len(str(top_k[0])) + 1
        min_len, max_len = desired_length, desired_length + 1
        assert len(predict_types) == 1

    for pred in predict_types:
        disabled_list = []
        for r in range(min_len, max_len + 1):
            for combo in itertools.combinations(layers, r):
                disabled = "".join(str(i) for i in combo)
                if top_k == []:
                    print(
                        f"python3 ${{python_files_dir}}/write_eval.py "
                        f"--model_size eightlayer --continuation 0 --ao 1 "
                        f"--predict_{pred} 1 --disabled_layers {disabled}"
                    )
                if disabled[:-1] in top_k: disabled_list.append(disabled)
    if len(predict_types) == 1: return(disabled_list)


def write_latex_tables(loss, block = True, round = 3):
    if block: filepath = f"iclr/proper/statistics_{loss}.csv"
    else: filepath = f"iclr/proper/statistics_{loss}_noblock.csv"
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    data = rows[1:]

    header.pop(8)
    header.pop(1)

    header = ["$\\nbins$", "$\Brier \downarrow$", "$\RPS$", "$\CE$", "$\HHI \downarrow$", "$\softmax \downarrow$", "$\softmax (\pm1) \downarrow$", "Accuracy \downarrow"]

    latex_lines = [" & ".join(header) + r" \\"]

    for row in data:
        row.pop(8)
        row.pop(1)
        row_round = [row[0]] + [ f'%.{round}f' % float(elem) for idx, elem in enumerate(row) if idx > 0 ]
        latex_lines.append(" & ".join(row_round) + r" \\")

    for idx, line in enumerate(latex_lines):
        if idx == 1:
            print("\hline")
        print(line)


if __name__ == "__main__":
    # three = commands(predict_types=["x"], top_k=[str(123),str(235),str(467)])
    write_latex_tables("rps")