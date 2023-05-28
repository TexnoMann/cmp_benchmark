import pickle
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--input", type=str, required=True,
                    help="Input pickle file for ploting graph")

algorithm_names = {'PJ': 'CBiRRT', 'TB': 'TBRRT', 'AT': 'AtlasRRT'}

def plot_field_comparison(data, title: str, field: str):
    _fig = plt.figure()
    exec_time_compare = []
    space_title_compare = []
    for space in data:
        time_results = data[space][field].tolist()
        exec_time_compare.append(time_results)
        space_title_compare.append(algorithm_names[space])
    plt.title(title)
    plt.grid(axis='y', color='0.95')
    plt.grid(axis='x', color='0.95')
    plt.boxplot(exec_time_compare, patch_artist=True, labels=space_title_compare)
    plt.show()

def main():
    args = parser.parse_args()
    print("Opening: {}".format(args.input))
    with open(args.input, 'rb') as file:
        data_plan = pickle.load(file)
    plot_field_comparison(data_plan, "Время планирования пути. RRTConnect", 'exec_time')

        
        

if __name__ == "__main__":
    main()