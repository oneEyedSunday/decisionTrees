import pandas as pd
import numpy as np
from pprint import pprint

dataset = pd.read_csv('datasets/zoo.data.csv',
                      names=['animal_name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator',
                             'toothed', 'backbone',
                             'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'class'])
dataset.drop('animal_name', axis=1)


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    _entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return _entropy


def info_gain(data, split_attribute_name, target_name='class'):
    total_entropy = entropy(data[split_attribute_name])
    values, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighed_entropy = np.sum([(counts[i] / np.sum(counts)) *
                              entropy(data.where(data[split_attribute_name] == values[i]).dropna()[target_name]) for i
                              in range(len(values))])
    return total_entropy - weighed_entropy


def id3(data, originaldata, features, target_attribute_name='class', parent_node_class=None):
    """

    :param data: the data for which the ID3 algorithm should be run --> In the first run this equals the total dataset
    :param originaldata: This is the original dataset needed to calculate the mode target feature value of the original dataset
                         in the case the dataset delivered by the first parameter is empty
    :param features: the feature space of the dataset . This is needed for the recursive call since during the tree growing process
                     we have to remove features from our dataset --> Splitting at each node
    :param target_attribute_name: the name of the target attribute
    :param parent_node_class: This is the value or class of the mode target feature value of the parent node for a specific node. This is
    also needed for the recursive call since if the splitting leads to a situation that there are no more features left in the feature
    space, we want to return the mode target feature value of the direct parent node.
    :return:

    1. We need to define stopping criteria, if met, we want to return a leaf node
    """

    # If all target_values have the same value, return this value
    # TODO(oneeyedsunday) shouldnt this be len(unique) == len(items)?
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(
            np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    # If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    # the mode target feature value is stored in the parent_node_class variable.
    elif len(features) == 0:
        return parent_node_class

    # If none of the above holds true, grow the tree!
    else:
        # Set the default value for this node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],
                                                                                       return_counts=True)[1])]

        # Select the feature which best splits the dataset
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]

        # Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create the tree structure. The root gets the name of the feature (best_feature)
        # with the maximum information gain in 1st run
        tree = {best_feature: {}}

        # Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]

        # Grow a branch under the root node for each possible value of the root node feature
        for value in np.unique(data[best_feature]):
            # Split the dataset along the value of the feature with the largest information gain
            # and therewith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()
            # Call the ID3 algorithm for each of those sub_datasets with the new parameters
            # Here the recursion comes in!
            subtree = id3(sub_data, dataset, features, target_attribute_name, parent_node_class)
            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
        return tree


def predict(query, tree, default=1):
    """
        Prediction of a new/unseen query instance. This takes two parameters:
        1. The query instance as a dictionary of the shape {"feature_name":feature_value,...}
        2. The tree
        we wander down the tree and check if we have reached a leaf or if we are still in a sub tree.
    """
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def train_test_split(_dataset):
    _training_data = _dataset.iloc[:80].reset_index(drop=True)
    _testing_data = _dataset.iloc[:80].reset_index(drop=True)
    return _training_data, _testing_data


def test(data, _tree):
    queries = data.iloc[:, :-1].to_dict(orient='records')
    predicted = pd.DataFrame(columns=["predicted"])

    for i in range(len(data)):
        predicted.loc[i, 'predicted'] = predict(queries[i], _tree, 1.0)
    print('The prediction accuracy is: ', (np.sum(predicted['predicted'] == data['class'])/len(data)) * 100, '%')


training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1]
"""
Train the tree, Print the tree and predict the accuracy
"""
tree = id3(training_data, training_data, training_data.columns[:-1])
pprint(tree)
test(testing_data, tree)
