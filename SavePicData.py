import matplotlib.pyplot as plt
def savepicdata(dataname='DDI',modelname='EnhancedAttentionMLPFusionModel', epochs=25,
                train_losses_history=[], train_f1_history=[], train_accuracy_history=[], train_precision_history=[], train_recall_history=[],
                dev_losses_history=[], dev_f1_history=[], dev_accuracy_history=[], dev_precision_history=[], dev_recall_history=[],
                test_losses_history=[], test_f1_history=[], test_accuracy_history=[], test_precision_history=[], test_recall_history=[],
                ):
    # 绘制训练集训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses_history, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(dataname+modelname+'Training Loss History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_f1_history, 'g-', label='Training F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Train f1 Score')
    plt.title(dataname+modelname+'Training F1 Score History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(dataname+modelname+'training_history.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 绘制验证集训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(dev_losses_history, 'b-', label='Development Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(dataname+modelname+'Development Loss History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(dev_f1_history, 'g-', label='Development F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Development f1 Score')
    plt.title(dataname+modelname+'Development F1 Score History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(dataname+modelname+'development_history.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # # 绘制测试集训练历史
    # plt.figure(figsize=(12, 4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(test_losses_history, 'b-', label='Test Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title(dataname+modelname+'Test Loss History')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(test_f1_history, 'g-', label='Test F1 Score')
    # plt.xlabel('Epoch')
    # plt.ylabel('Test f1 Score')
    # plt.title(dataname+modelname+'Test F1 Score History')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.savefig(dataname+modelname+'test_history.png', dpi=300, bbox_inches='tight')
    # plt.show()

    print(f"\n{dataname}{modelname}图片生成保存完成")