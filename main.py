# Current project dependencies
import env


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
    print("\t9. Run Bot (paper trading)")
    print("\t0. Exit")

    choice = input("\nEnter your choice: ")
    choice = int(choice) if choice != "" else 99

    if choice == 1:
        print("Downloading data...")
        # Current project dependencies
        import picture_taker as pt

        pt.take_picture()
        print("\tDone.")
    elif choice == 2:
        print("Combining data...")
        # Current project dependencies
        import file_combiner as fc

        fc.combine_all_tickers(from_data_api=True)
        print("\tDone.")
    elif choice == 3:
        print("Calculating trade outcomes...")
        # Current project dependencies
        import training_data_cooker as tdc

        tdc.calculate_all_trade_outcomes_to_dataframe()
        print("\tDone.")
    elif choice == 4:
        print("Creating data vectors...")
        # Current project dependencies
        import data_vectorizer as dv

        dv.create_training()
        print("\tDone.")
    elif choice == 5:
        print("Scaling data...")
        # Current project dependencies
        import data_vectorizer as dv

        dv.scale_all_data()
        print("\tDone.")
    elif choice == 6:
        print("Creating training data...")
        # Current project dependencies
        import data_vectorizer as dv

        dv.create_labels_for_all_bars()
        print("\tDone.")
    elif choice == 7:
        print("Start training...")
        # Current project dependencies
        import classic_classifier as cc

        cc.train_classifier()
        print("\tDone.")
    elif choice == 8:
        print("Start training...")
        # Current project dependencies
        import cnn_classifier as cnc

        cnc.train()
        print("\tDone.")
    elif choice == 9:
        print("Start paper trading...")
        # Current project dependencies
        import paper_life_simulator as pls

        paper_trader = pls.PaperTrader()
        paper_trader.run()
        print("Stopped paper trading.")
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
