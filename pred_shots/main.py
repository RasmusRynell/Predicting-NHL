from handler import *
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
            player_ids = [8471214]#, 8475167, 8478463, 8475744, 8477504, 8477492, 8480839, 8477499, 8476881]
            for player_id in player_ids:
                data = generate_csv(player_id)
                save_csv(data, str(player_id) + ".csv")


        elif arg_in.lower() == 'pred':
            bets = get_bets()

            # Remove all but last 5 bets in dict
            keys = list(bets.keys())
            new_bets = {}
            for i in range(10, 15):
                new_bets[keys[i]] = bets[keys[i]]


            predictions = predict_games(new_bets)

            print("Predictions saved to file")
            print("Predictions: ")
            #print(json.dumps(predictions, indent=4))

            # Save predictions dict to a file
            with open('./external/predictions/test.json', 'w') as f:
                json.dump(predictions, f)
                



        elif arg_in.lower() == 'eval':
            bets = {}
            # Read in bets from file
            with open('./external/predictions/test.json', 'r') as f:
                bets = json.load(f)
            evaluate_bets(bets)
            


        else:
            print(f"\"{arg_in}\" is no command")