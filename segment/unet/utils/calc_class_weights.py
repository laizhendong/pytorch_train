

def calc_class_weights(class_size,class_present):
    num_classes = len(class_size)
    freq, class_weight = ([0 for _ in range(num_classes)] for _ in range(2))
    median_freq = 0
    for l in range(num_classes):
        if class_present[l] == 0:
            raise Exception("The class {} is not present in the dataset".format(l+1))

        freq[l] = float(class_size[l]) / float(class_present[l])  # calculate freq per class
        median_freq = 0.5*sum(freq)/(num_classes)

    for c in range(num_classes):
        class_weight[c] = float(median_freq) / float(freq[c])

    return class_weight