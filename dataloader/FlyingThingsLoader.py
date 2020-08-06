from python_pfm import readPFM

def load_disparity(file_path):
    return readPFM(file_path)

if __name__ == '__main__':
    print(load_disparity("sample_dataset/disparity/0006.pfm"))

