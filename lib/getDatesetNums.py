from pathlib import Path


def box_count_yolo(path, num_classes):
    lis = [0 for _ in range(num_classes)]
    n = 0
    for f in Path.iterdir(Path(path)):
        with open(Path(f)) as file:
            temp = file.read()
            temp = temp.split()
            for num in temp:
                if len(str(num)) == 1:
                    lis[int(num)] += 1
            print(lis)
        n += 1
    print(f"--------{n} files finished--------")
    print(lis)


if __name__ == '__main__':
    box_count_yolo("/VOCdevkit/labels", 2)  # dataset path and nums of classes