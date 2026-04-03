from datasets.base_dataset import BaseSoilDataset

def main():
    print("Soil HSI Project Starting...")

if __name__ == "__main__":
    main()


from datasets.karlsruhe import KarlsruheDataset

def main():
    dataset = KarlsruheDataset()
    dataset.load()

    data = dataset.get_data()

    print(data["spectra"].shape)

if __name__ == "__main__":
    main()
