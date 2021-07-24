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
            player_ids = [8471214, 8475167, 8478463, 8475744, 8477504, 8477492, 8480839, 8477499, 8476881]
            for player_id in player_ids:
                data, game_date = generate_csv(player_id)
                game_date = str(game_date).replace(" ", "-").replace(":", "-")
                save_csv(data, str(player_id) + "_" + str(game_date) + ".csv")


        elif arg_in.lower() == 'eval':
            # Feature extraction / selection
            # From file or from "input"
            extract_and_select_var = {}
            with open('./external/configs/test.cfg') as f:
                extract_and_select_var = json.loads(f.read())
            save_csv(evaluate_setup(extract_and_select_var), extract_and_select_var['save_path'])

        elif arg_in.lower() == 'pred':
            games = [
            {
                'player_id': 8471214,
                'game_id': '2020020730',
                'date': '2021-04-22',
                'target': ['3.5', '4.5'],
                'config': "./external/configs/default.cfg"
            },
            {
                'player_id': 8475167,
                'game_id': '2020020048',
                'date': "2021-05-07",
                'target': ['2.5'],
                'config': "./external/configs/default.cfg"
            },
            {
                'player_id': 8478463,
                'game_id': '2020020839',
                'date': "2021-05-24",
                'target': ['2.5'],
                'config': "./external/configs/default.cfg"
            },
            {
                'player_id': 8475744,
                'game_id': '2020020814',
                'date': '2021-05-03',
                'target': ['1.5'],
                'config': "./external/configs/default.cfg"
            },
            {
                'player_id': 8477504,
                'game_id': '2020020834',
                'date': '2021-05-05',
                'target': ['1.5'],
                'config': "./external/configs/default.cfg"
            },
            {
                'player_id': 8477492,
                'game_id': '2020030154',
                'date': '2021-05-22',
                'target': ['3.5'],
                'config': "./external/configs/default.cfg"
            },
            {
                'player_id': 8476881,
                'game_id': '2020020162',
                'date': '2021-05-12',
                'target': ['2.5'],
                'config': "./external/configs/default.cfg"
            },
            {
                'player_id': 8477499,
                'game_id': '2020020842',
                'date': '2021-05-06',
                'target': ['1.5'],
                'config': "./external/configs/default.cfg"
            },
            {
                'player_id': 8476881,
                'game_id': '2020020162',
                'date': '2021-05-12',
                'target': ['2.5'],
                'config': "./external/configs/default.cfg"
            }
            ]

            predictions = predict_games(games)
            
            # Save predictions dict to a file
            with open('./external/predictions/test.json', 'w') as f:
                json.dump(predictions, f)
                
            print("Predictions saved to file")
            print("Predictions: ")
            print(json.dumps(predictions, indent=4))


        else:
            print(f"\"{arg_in}\" is no command")