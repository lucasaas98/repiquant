# Current project dependencies
import data_vectorizer as dv
import file_combiner as fc
import picture_taker as pt
import training_data_cooker as tdc


def main():
    print("Repiquant at your service.")
    print("\t1. Download missing data")
    print("\t2. Combine data files")
    print("\t3. Calculate trade outcomes")
    print("\t4. Scale data")
    print("\t5. Create training data")
    print("\t0. Exit")

    choice = int(input("\nEnter your choice: "))

    if choice == 1:
        pt.take_picture()
    elif choice == 2:
        fc.combine_all_tickers(from_data_api=True)
    elif choice == 3:
        tdc.calculate_all_trade_outcomes_to_dataframe()
    elif choice == 4:
        dv.scale_all_data()
    elif choice == 5:
        dv.create_labels_for_all_bars()
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
