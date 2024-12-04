# Current project dependencies
import classic_classifier as cc
import cnn_classifier as cnc
import data_vectorizer as dv
import file_combiner as fc
import picture_taker as pt
import training_data_cooker as tdc


def main():
    print("Repiquant at your service.")
    print("\t1. Download missing data")
    print("\t2. Combine data files")
    print("\t3. Calculate trade outcomes")
    print("\t4. Create Data Vectors")
    print("\t5. Scale data")
    print("\t6. Create training data")
    print("\t7. Train (interval:5min)")
    print("\t8. Train CNN (interval:5min)")
    print("\t0. Exit")

    choice = int(input("\nEnter your choice: "))

    if choice == 1:
        print("Downloading data...")
        pt.take_picture()
        print("\tDone.")
    elif choice == 2:
        print("Combining data...")
        fc.combine_all_tickers(from_data_api=True)
        print("\tDone.")
    elif choice == 3:
        print("Calculating trade outcomes...")
        tdc.calculate_all_trade_outcomes_to_dataframe()
        print("\tDone.")
    elif choice == 4:
        print("Creating data vectors...")
        dv.create_training()
        print("\tDone.")
    elif choice == 5:
        print("Scaling data...")
        dv.scale_all_data()
        print("\tDone.")
    elif choice == 6:
        print("Creating training data...")
        dv.create_labels_for_all_bars()
        print("\tDone.")
    elif choice == 7:
        print("Start training...")
        cc.train_classifier()
        print("\tDone.")
    elif choice == 8:
        print("Start training...")
        cnc.train()
        print("\tDone.")
    elif choice == 0:
        return False
    else:
        print("Invalid choice")

    print()
    return True


if __name__ == "__main__":
    flag = True
    while flag:
        flag = main()
