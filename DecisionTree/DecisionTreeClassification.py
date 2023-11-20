import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from graphviz import Digraph
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class DecisionTree:
    def __init__(self, feature):
        self.samples = None
        self.accuracy = None
        self.feature = feature
        self.child = {}

    def visualize_tree(self, dot=None, parent_color='black'):
        if dot is None:
            dot = Digraph(comment='Decision Tree', format='png',
                          graph_attr={'rankdir': 'TB', 'bgcolor': '#AFDBF5', 'dpi': '300'})

        label = f"{self.feature}\nSamples: {self.samples}" if self.samples is not None else self.feature
        label += f"\nAccuracy: {self.accuracy}%" if self.accuracy is not None else ""

        if self.feature.startswith('class='):
            dot.node(str(id(self)), label, shape='box', style='filled', fillcolor='#E1AD01',
                     fontcolor='black', fontsize='12', fontname='Serif', width='0', height='0',
                     fixedsize='false', margin='0.08', peripheries='1', penwidth='2.0',
                     **{'pencolor': 'opaque'})  # transparent
        else:
            dot.node(str(id(self)), label, shape='box', style='filled', fillcolor='#478778',
                     fontcolor='black', fontsize='12', fontname='Serif', width='0', height='0',
                     fixedsize='false', margin='0.08', peripheries='1', penwidth='2.0', **{'pencolor': 'opaque'})

        if self.child:
            for value, child_tree in self.child.items():
                child_tree.visualize_tree(dot, parent_color='black' if not self.feature.startswith(
                    'class=') else '#478778')
                # Edge styling for decision edges
                edge_color = 'orange' if not self.feature.startswith('class=') and child_tree.feature.startswith(
                    'class=') else parent_color
                dot.edge(str(id(self)), str(id(child_tree)), label=value,
                         color=edge_color, fontsize='10',
                         fontname='Serif', penwidth='2.0')

        return dot


class DecisionTreeClassification:
    def __init__(self, impurity_measure, parallelism):
        self.impurity_measure = impurity_measure
        self.parallelism = parallelism
        self.executor = ThreadPoolExecutor(max_workers=parallelism)
        self.impurity_calc = self._get_impurity_calc(impurity_measure)
        self.root = None

    def _segregate_dataframe(self, df, col_idx):
        categories = df[:, col_idx]
        unique_categories, category_indices = np.unique(categories, return_inverse=True)
        masks = (category_indices[:, None] == np.arange(len(unique_categories)))
        segregated_rows = {category: df[mask] for category, mask in zip(unique_categories, masks.T)}
        return segregated_rows, unique_categories, category_indices

    def _calculate_entropy(self, category_array):
        category_counts = Counter(category_array)
        total_samples = len(category_array)
        probabilities = np.array(list(category_counts.values())) / total_samples
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _gini_index(self, array):
        array = np.array(array).flatten()
        unique_elements, counts = np.unique(array, return_counts=True)
        probabilities = counts / len(array)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _entropy(self, array):
        array = np.array(array).flatten()
        unique_elements, counts = np.unique(array, return_counts=True)
        probabilities = counts / len(array)
        probabilities[probabilities == 0] = 1e-10
        entropy_val = -np.sum(probabilities * np.log2(probabilities))
        return entropy_val

    def _get_impurity_calc(self, impurity_measure):
        calc_map = {
            'gini': lambda array: self._gini_index(array),
            'entropy': lambda array: self._entropy(array)
        }
        return calc_map[impurity_measure]

    def _information_gain(self, args):
        HD, df, col_idx = args
        seg_df, categories, cat_idx = self._segregate_dataframe(df, col_idx)
        h_map = {}
        for attr, sdf in seg_df.items():
            h_map[attr] = self.impurity_calc(sdf[:, -1])
        counts = np.bincount(cat_idx)
        probs = counts / df.shape[0]
        HDv = np.vectorize(lambda x: h_map.get(x, 0.0))(categories)
        IG = HD - np.sum(probs * HDv)
        return IG

    def _calculate_gain(self, df, HD, tasks):
        futures = [self.executor.submit(self._information_gain, (HD, df, i)) for i in tasks]

        result = np.zeros(len(tasks))
        for future in as_completed(futures):
            index = futures.index(future)
            result[index] = future.result()

        return result

    def _build_tree(self, df, header):
        HD = self.impurity_calc(df[:, -1])
        if HD == 0.0:
            root = DecisionTree(feature='class=' + df[0, -1])
            root.samples, root.accuracy, root.child = df[:, 0].flatten().shape[0], 100, {}
            return root
        if len(header) < 2:
            unique_values, counts = np.unique(df[:, 0].flatten(), return_counts=True)
            max_occurrence_index = np.argmax(counts)
            root = DecisionTree(feature='class=' + unique_values[max_occurrence_index])
            root.samples, root.accuracy, root.child = df[:, 0].flatten().shape[0], (
                    counts[max_occurrence_index] / df[:, 0].flatten().shape[0]) * 100, {}
            return root
        tasks = range(df.shape[1] - 1)
        result = self._calculate_gain(df, HD, tasks)
        max_idx = np.argmax(result)
        root = DecisionTree(header[max_idx])
        root.samples = df[:, -1].shape[0]
        segregated_rows, unique_categories, category_indices = self._segregate_dataframe(df, max_idx)
        for key, df in segregated_rows.items():
            segregated_rows[key] = np.delete(segregated_rows[key], max_idx, axis=1)
        sub_header = np.delete(header, max_idx)
        for category in unique_categories:
            root.child[category] = self._build_tree(segregated_rows[category], sub_header)
        return root

    def _predict_each(self, X_test, root):
        if root.feature.startswith('class='):
            return root.feature.split('=')[1]
        category = X_test[root.feature]
        if category is not None and category in root.child:
            return self._predict_each(X_test, root.child[category])
        else:
            return ''

    def _create_list_of_dicts(self, X_test, header):
        return [{header[i]: row[i] for i in range(len(header) - 1)} for row in X_test]

    def fit(self, df, header):
        self.root = self._build_tree(df, header)
        return self.root

    def predict(self, X_test):
        X_test = self._create_list_of_dicts(X_test, header)
        y_pred = np.array([self._predict_each(X, self.root) for X in X_test])
        return y_pred.reshape(y_pred.size, 1)


def csv_to_numpy(file_path):
    with open(file_path, 'r') as file:
        header = np.array(file.readline().strip().split(','))
    data = np.genfromtxt(file_path, delimiter=',', dtype=None, names=True, encoding=None)
    return header, np.array([list(row) for row in data])


# Example usage:
file_path = 'C:\code\MLsnipets\dicision-tree\data.csv'
header, df = csv_to_numpy(file_path)

train_data, test_data = train_test_split(df, test_size=0.4, random_state=42)
decision_tree = DecisionTreeClassification('entropy', 4)
result = decision_tree.fit(train_data, header)

X_test, y_test = test_data[:, :-1], test_data[:, -1]
y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# dot = result.visualize_tree()
# dot.render('decision_tree', format='png', cleanup=True)
