from handler import *


def print_help():
    print("Current commands: ")
    print("1. help: Shows this")
    print("2. exit: Exits the process, nothing is saved")



if __name__ == "__main__":
    while True:
        arg_in = input(">>> ")

        # Initial commands
        if arg_in.lower() == 'help' or arg_in == '1' or arg_in == 'h':
            print_help()

        elif arg_in.lower() == 'exit' or arg_in == '2' or arg_in == 'e':
            print("Exiting...")
            break

        
        # Database commands
        elif arg_in.lower() == 'eval':
            eval_bets()

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

        else:
            print(f"\"{arg_in}\" is no command")