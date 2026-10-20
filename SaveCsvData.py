#训练历史保存为csv
import csv
import os
def savecsvdata(filename='training_history.csv', epochs=25,
                train_losses_history=[], train_f1_history=[], train_accuracy_history=[], train_precision_history=[], train_recall_history=[],
                dev_losses_history=[], dev_f1_history=[], dev_accuracy_history=[], dev_precision_history=[], dev_recall_history=[],
                test_losses_history=[], test_f1_history=[], test_accuracy_history=[], test_precision_history=[], test_recall_history=[],
                ):
    # 定义CSV列名
    fieldnames = [
        '循环次数',
        'train_loss', 'dev_loss', #'test_loss',
        'train_f1', 'dev_f1', #'test_f1',
        'train_accuracy', 'dev_accuracy', #'test_accuracy',
        'train_precision', 'dev_precision', #'test_precision',
        'train_recall', 'dev_recall', #'test_recall'
    ]

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 逐行写入数据
        for i in range(epochs):
            row = {
                '循环次数': i + 1,
                'train_loss': train_losses_history[i],
                'dev_loss': dev_losses_history[i],
                #'test_loss': test_losses_history[i],
                'train_f1': train_f1_history[i],
                'dev_f1': dev_f1_history[i],
                #'test_f1': test_f1_history[i],
                'train_accuracy': train_accuracy_history[i],
                'dev_accuracy': dev_accuracy_history[i],
                #'test_accuracy': test_accuracy_history[i],
                'train_precision': train_precision_history[i],
                'dev_precision': dev_precision_history[i],
                #'test_precision': test_precision_history[i],
                'train_recall': train_recall_history[i],
                'dev_recall': dev_recall_history[i],
                #'test_recall': test_recall_history[i]
            }
            writer.writerow(row)
    print(f"训练历史已保存到: {os.path.abspath(filename)}")