from handler import *


def print_help():
    print("Current commands: ")
    print("1. help: Shows this")
    print("2. exit: Exits the process, nothing is saved")
    print("3. eval: ")
    print("4. und: Refreshes/Updates the local NHL database")
    print("5. and: Add nicknames to the database")
    print("6. ubd: Add \"old\" bets (from bookies that's located on a local file) to database")
    print("7. gen: Generate a CSV file containing all information for a player going back to 2017/09/15")
    print("8. pre: Perform preprocessing")




if __name__ == "__main__":
    while True:
        arg_in = input(">>> ")

        # Initial commands
        if arg_in.lower() == 'help' or arg_in == '1' or arg_in == 'h':
            print_help()

        elif arg_in.lower() == 'exit' or arg_in == '2' or arg_in == 'e':
            print("Exiting...")
            break

        #elif arg_in.lower() == 'eval':
        #    eval_bets()

        # Dev
        elif arg_in.lower() == 'und':
            update_nhl_db()
            print("Nhl database updated")

        elif arg_in.lower() == 'and':
            add_nicknames_nhl_db()
            print("Nicknames added")

        elif arg_in.lower() == 'ubd':
            print("What file do you want to add? (\"b\" for back) (EX: 2017-03-21.bet365) (Empty for all in folder)")
            f = input(">>> > ")
            if f == 'b':
                print("Back...")
            elif f == "":
                update_bets_db()
                print("Bets database updated")
            else:
                update_bets_db(f)
                print("Bets database updated")

        elif arg_in.lower() == 'gen':
            start_time = datetime(2017, 9, 15)
            end_time = datetime.now()
            print("Enter player id (\"b\" for back) (EX: 8475167)")
            player_id = 8475167#input(">>> > ")
            generate_csv(player_id)


        elif arg_in.lower() == 'eval':
            # Feature extraction / selection
            # From file or from "input"
            extract_and_select_var = {}
            with open('./external/preprocessing_configs/test.cfg') as f:
                extract_and_select_var = json.loads(f.read())
            save_csv(evaluate_setup(extract_and_select_var), extract_and_select_var['save_path'])


        else:
            print(f"\"{arg_in}\" is no command")