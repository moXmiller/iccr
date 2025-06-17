def retrieve_run_id(args, lin_reg = False):
    if lin_reg: raise NotImplementedError
    model_size = args.model_size
    data_arg = args.data
    ao = args.ao
    transformation = args.transformation
    if data_arg == "gaussian":
        if not ao:
            if model_size == "tiny":
                cont_run_id = "7cf75f87-3e4e-4182-ab04-961f18d54063"
                cf_run_id = "c75ffbdf-7927-4293-84dd-dc7147bc6a54"
            elif model_size == "small":
                cont_run_id = "270cbda4-12d1-4648-80f4-2de1a44fea1a"
                cf_run_id = "87d1196d-b6e6-4b6b-8bdf-47ca95f8d4ee"
            elif model_size == "standard":
                cont_run_id = "f4097f40-065a-45c6-ad54-b445938fb58c"
                cf_run_id = "d63843a7-ae65-4789-9877-b40b7fcefc9e"
            elif model_size == "fourlayer":
                cont_run_id = None
                cf_run_id = "3c38abad-34d0-4f44-b4c6-ec1a0e44c2af"
            elif model_size == "threelayer":
                cont_run_id = None
                cf_run_id = "aaf6c5c5-4de8-4340-bc2a-4360e2151759"
            elif model_size == "twolayer":
                cont_run_id = None
                cf_run_id = "f200b29e-42ef-42f9-a88b-9fcf8a9434b8"
            elif model_size == "onelayer":
                cont_run_id = None
                cf_run_id = "c9fee134-2560-4e7d-8ce3-a2c470afc201"
            elif model_size == "eightlayer":
                if transformation == "addlin":
                    cont_run_id = None
                    cf_run_id = "5983f254-6be7-45cd-b711-955d7f52081d"
                elif transformation == "mullin":
                    cont_run_id = None
                    cf_run_id = "0b8fe81c-507e-45e9-ad2f-5fbdcb78a3c5"
                elif transformation == "tanh":
                    cont_run_id = None
                    cf_run_id = "4eeb18e8-e9e1-4c16-b3f7-8c22ae3dcd3d"
                elif transformation == "sigmoid":
                    cont_run_id = None
                    cf_run_id = "fc479a29-1462-4908-958b-95b65d72122f"
            elif model_size == "2h_4l":
                cont_run_id = None
                cf_run_id = "b51436de-5d31-4a17-a2e4-a533932592c7"
            elif model_size == "4h_2l":
                cont_run_id = None
                cf_run_id = "1562867a-d268-4f01-8939-a0a6a02b35dd"
            elif model_size == "eighthead":
                cont_run_id = None
                cf_run_id = "fa58bafd-01c3-406d-a1ec-6691fd7497aa"
            elif model_size == "fourlayer4vars":
                cont_run_id = None
                cf_run_id = "ef7e91bd-13ea-4828-a403-f366c6dc77ff"
            elif model_size == "fourlayer3vars":
                cont_run_id = None
                cf_run_id = "80c8a759-8b83-4fb5-afa0-af95c6a1f587"
            elif model_size == "fourlayer2vars":
                cont_run_id = None
                cf_run_id = "503c8471-75dc-4cdb-9d67-2808427ef1ac"
            elif model_size == "eightlayer4vars":
                cont_run_id = None
                cf_run_id = "a68abefd-4f5e-491b-a9ec-36ff294e3fb6"
            elif model_size == "eightlayer3vars":
                cont_run_id = None
                cf_run_id = "8c8e5b13-e99b-4aeb-a172-48b13ce43848"
            elif model_size == "eightlayer2vars":
                cont_run_id = None
                cf_run_id = "2b0c0294-22f3-479d-87a9-e1b1cb266d21"
            elif model_size == "rnn":
                cont_run_id = None
                cf_run_id = "f95475f9-2a3d-4124-9fb0-b86068dce700"
            elif model_size == "mlponly":
                cont_run_id = None
                cf_run_id = "edf1da35-e8ef-4ea9-9183-ccf7447e4f97"
            elif model_size == "lstm_128_2":
                cont_run_id = None
                cf_run_id = "e49d8285-7e7a-45f1-98ec-6c7b040293b1"
            elif model_size == "lstm_128_3":
                cont_run_id = None
                cf_run_id = "b69a46ab-054e-4e68-a706-d5d30edc0c14"
            elif model_size == "lstm_256_2":
                cont_run_id = None
                cf_run_id = "f00c515b-92d7-4692-b321-eb37af1709ca"
            elif model_size == "lstm_256_3":
                cont_run_id = None
                cf_run_id = "0d81b0b6-2929-4bc0-b258-dfd2086626ff"
            elif model_size == "gru_128_2":
                cont_run_id = None
                cf_run_id = "0b3015f8-8680-41b5-a41c-80225da440bf"
            elif model_size == "gru_128_3":
                cont_run_id = None
                cf_run_id = "5a53c23d-1b57-4b32-802c-45e8158d0a82"
            elif model_size == "gru_256_2":
                cont_run_id = None
                cf_run_id = "7c8378f1-ba6c-465a-bf18-0287fd730cff"
            elif model_size == "gru_256_3":
                cont_run_id = None
                cf_run_id = "670f5882-7b06-4b6e-b747-37f9695ef82e"
            elif model_size == "rnn_128_2":
                cont_run_id = None
                cf_run_id = "ccf7ca34-42dd-4612-83fa-730804fa94b2"
            elif model_size == "rnn_256_2":
                cont_run_id = None
                cf_run_id = "36db4050-029f-4278-af8a-699df7079898"
            elif model_size == "rnn_128_3":
                cont_run_id = None
                cf_run_id = "d3d08c47-c663-417e-9a71-cc431c12cba7"
            elif model_size == "rnn_256_3":
                cont_run_id = None
                cf_run_id = "c0221c67-cd21-4eba-9dc8-6a4852fe7c30"
            elif model_size == "cf_u_1":
                cont_run_id = None
                cf_run_id = "708a4318-7407-41a9-80fc-1fa553bc7d52"
            elif model_size == "cf_u_3":
                cont_run_id = None
                cf_run_id = "5c92d6af-1248-4783-b38f-2ee249edf0e8"
            elif model_size == "cf_u_5":
                cont_run_id = None
                cf_run_id = "96314b1d-ad48-4e97-9092-242cea17348f"
            elif model_size == "cf_u_10":
                cont_run_id = None
                cf_run_id = "7a021f66-5d86-4360-81e7-1c8b7dae4352"
            elif model_size == "cf_u_15":
                cont_run_id = None
                cf_run_id = "2855286b-7540-4c0a-965d-09d6bfd07d74"
            elif model_size == "cf_u_20":
                cont_run_id = None
                cf_run_id = "9b412de3-73db-4385-9019-2937d034f87f"
            elif model_size == "cf_u_30":
                cont_run_id = None
                cf_run_id = "efc6f7e1-e155-4637-84d9-d3d36e000b3d"
            elif model_size == "cf_u_50":
                cont_run_id = None
                cf_run_id = "64f5cf3b-c768-460c-a805-67e080f55760"
            elif model_size == "cf_u_75":
                cont_run_id = None
                cf_run_id = "a8ee6ed2-9a64-4c94-869a-0091dde29865"
            elif model_size == "cf_u_100":
                cont_run_id = None
                cf_run_id = "d33a94b1-4e5f-481c-b572-6fde0baa1e03"
            elif model_size == "cf_u_300":
                cont_run_id = None
                cf_run_id = "6ac2c6a8-77fc-4da3-a63a-04f664c945ff"
            elif model_size == "cf_u_500":
                cont_run_id = None
                cf_run_id = "0f88efe3-a571-4341-a966-e7bcf2b8ab39"
            elif model_size == "cf_u_750":
                cont_run_id = None
                cf_run_id = "0efb64f7-0273-4437-8469-7bcc67e3d1d1"
            elif model_size == "cf_u_1000":
                cont_run_id = None
                cf_run_id = "248fa759-9b05-431c-bbdd-e0d9238c5d32"
            elif model_size == "cf_u_1500":
                cont_run_id = None
                cf_run_id = "923b4d8e-9dce-480a-91c1-7fd18b8c9581"
            elif model_size == "cf_n_1":
                cont_run_id = None
                cf_run_id = "2dcd0af5-c3e4-43f2-82af-25d83dd584d7"
            elif model_size == "cf_n_3":
                cont_run_id = None
                cf_run_id = "85804e9e-707f-460f-9a92-b68e06e453fd"
            elif model_size == "cf_n_5":
                cont_run_id = None
                cf_run_id = "224dd945-46cc-4718-a0a0-f64236d51b0b"
            elif model_size == "cf_n_10":
                cont_run_id = None
                cf_run_id = "85fab958-c149-4000-942a-daca6c4244b8"
            elif model_size == "cf_n_15":
                cont_run_id = None
                cf_run_id = "d44517dc-99cf-46ab-add9-d971c24fe6df"
            elif model_size == "cf_n_20":
                cont_run_id = None
                cf_run_id = "bda31f97-0c18-4be9-9c8c-84a50b70ad83"
            elif model_size == "cf_n_30":
                cont_run_id = None
                cf_run_id = "357f7fee-4190-4fc3-ab0f-540d9b7a4043"
            elif model_size == "cf_n_50":
                cont_run_id = None
                cf_run_id = "0615133c-8a70-4b0e-9fe7-d4d9f0ddbe8b"
            elif model_size == "cf_n_75":
                cont_run_id = None
                cf_run_id = "7cea77da-f4e1-456f-84cd-281ca9e0c5d3"
            elif model_size == "cf_n_100":
                cont_run_id = None
                cf_run_id = "22030f81-f890-4e6b-a24a-2d35553a7bca"
            elif model_size == "cf_n_300":
                cont_run_id = None
                cf_run_id = "36bbb501-1390-4dc3-b531-227317736a9c"
            elif model_size == "cf_n_500":
                cont_run_id = None
                cf_run_id = "642abde6-f4e6-4efc-b8a7-7c49de94cd6e"
            elif model_size == "cf_n_1000":
                cont_run_id = None
                cf_run_id = "a324f0a2-7c2e-4882-885a-426f70e68d31"
            elif model_size == "cf_n_1500":
                cont_run_id = None
                cf_run_id = "c34bcd5a-9ebd-4cbc-aac8-fb0a64d8db25"
            else: raise NotImplementedError
        else:
            if model_size == "tiny":
                cont_run_id = "abd1cae7-dfec-40ed-b841-14caf670720a"
                cf_run_id = "275c1fac-f726-4eeb-9d0e-ceb824d171f4"
            elif model_size == "small":
                cont_run_id = "63288af2-c27a-40f6-9d5c-ce56ea60a049"
                cf_run_id = "478fe3cc-09e2-4db1-8562-af29c8a2d586"
            elif model_size == "standard":
                cont_run_id = None
                cf_run_id = "ae36f1cf-68c0-4f19-8115-5e4a863edbb6"
            elif model_size == "fourlayer":
                cont_run_id = None
                cf_run_id = "948e2109-b969-44e3-915c-f82c7e24e8e2"
            elif model_size == "onelayer":
                cont_run_id = None
                cf_run_id = "6bab095b-e9f6-4d24-86d0-ed4bb705db2a"
            elif model_size == "twelvelayer":
                cont_run_id = None
                cf_run_id = "8719dfc8-866f-400e-b84e-cfa08b54f0e9"
            elif model_size == "sixteenlayer":
                cont_run_id = None
                cf_run_id = "1cf57f3c-7423-4e5d-ab96-4d05f011ead6"
            elif model_size == "twolayer":
                cont_run_id = None
                cf_run_id = "e3257fcf-4758-4bea-89f2-619dc598a24d"
            elif model_size == "threelayer":
                cont_run_id = None
                cf_run_id = "48018e24-e548-47e9-a0ac-8692318b766c"
            elif model_size == "eightlayer":
                if transformation == "addlin":
                    cont_run_id = None
                    cf_run_id = "5c2da617-f003-4a96-b863-f9c1323e9d31"
                elif transformation == "mullin":
                    cont_run_id = None
                    cf_run_id = "86b75832-4642-4ce9-a795-b25d20b32eb8"
                elif transformation == "tanh":
                    cont_run_id = None
                    cf_run_id = "5e0b50d2-b47d-41b0-9b8a-64606ac5bd7c"
                elif transformation == "sigmoid":
                    cont_run_id = None
                    cf_run_id = "d2d26073-9b2a-4ed0-bcc7-d5057a3ed6b5"
            elif model_size == "one_mlp":
                cont_run_id = None
                cf_run_id = "60cdb3e1-26e3-4e1c-94ca-74612af07779"
            elif model_size == "2h_4l":
                cont_run_id = None
                cf_run_id = "2594e6da-76b5-4e01-bb24-2b76259f68a8"
            elif model_size == "4h_2l":
                cont_run_id = None
                cf_run_id = "450aaff1-8470-4c8b-b5f0-dc3c494524ba"
            elif model_size == "eighthead":
                cont_run_id = None
                cf_run_id = '3b1579b3-d8c2-4063-ba56-4a9d702c0809'
            else: raise NotImplementedError
    elif data_arg == "sde":
        if not ao:
            if model_size == "standard":
                cont_run_id = None
                cf_run_id = "d4ecf0e5-b6b7-4716-8a02-a4e0d503a6be"
            elif model_size == "eightlayer":
                cont_run_id = None
                cf_run_id = "b5bc0b90-5cb1-4b60-90d7-230d97901b72"
            elif model_size == "gru":
                cont_run_id = None
                cf_run_id = "b96556a9-144a-49cb-b67a-d7686bd79441"
            elif model_size == "lstm":
                cont_run_id = None
                cf_run_id = "5047ea52-e64b-4075-b0c6-9f4188525850"
            elif model_size == "rnn":
                cont_run_id = None
                cf_run_id = "6e004f8e-b303-4734-8dee-a4a0926ebf4a"
        else:
            if model_size == "standard":
                cont_run_id = None
                cf_run_id = "dc8ebeb2-1601-424c-a9cd-bd8f1e7b3be6"
            if model_size == "eightlayer":
                cont_run_id = None
                cf_run_id = "c8e4f569-7857-4548-bbf6-c1f781c285c0"
                
    return cont_run_id, cf_run_id
