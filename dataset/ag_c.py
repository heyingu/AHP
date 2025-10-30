import csv
from datasets import load_dataset

def convert_agnews_to_tsv():
    # 加载AG News数据集
    raw_dset = load_dataset("ag_news")
    
    # 处理每个分割（train, test, validation）
    for split, dset in raw_dset.items():
        output_file = f"ag_news_{split}.tsv"
        
        with open(output_file, "w", newline="", encoding="utf-8") as tsvfile:
            # 创建TSV写入器
            writer = csv.writer(tsvfile, delimiter="\t", quotechar='"', 
                              quoting=csv.QUOTE_MINIMAL)
            
            # 写入标题行（可选）
            writer.writerow(["text", "label"])
            
            # 遍历数据集中的每个样本
            for example in dset:
                text = example["text"]
                label = example["label"]
                
                # 写入TSV文件
                writer.writerow([text, label])
        
        print(f"已保存 {split} 分割到 {output_file}, 包含 {len(dset)} 个样本")

# 运行转换函数
if __name__ == "__main__":
    convert_agnews_to_tsv()