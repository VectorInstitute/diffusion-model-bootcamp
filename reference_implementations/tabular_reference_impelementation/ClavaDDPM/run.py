import os

from complex_pipeline import *
from gen_single_report import gen_single_report

if __name__ == '__main__':
    config_path = 'configs/movie_lens.json'
    configs, save_dir = load_configs(config_path)
    tables, relation_order, dataset_meta = load_multi_table(configs['general']['data_dir'])
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, save_dir, configs)
    tables, models = clava_training(tables, relation_order, save_dir, configs)
    cleaned_tables, synthesizing_time_spent, matching_time_spent = clava_synthesizing(
        tables, 
        relation_order, 
        save_dir, 
        all_group_lengths_prob_dicts, 
        models,
        configs,
        sample_scale=1 if not 'debug' in configs else configs['debug']['sample_scale']
    )
    report = clava_eval(tables, save_dir, configs, relation_order, cleaned_tables)
    test_tables, _, _ = load_multi_table(configs['general']['test_data_dir'])
    real_tables, _, _ = load_multi_table(configs['general']['data_dir'])

    for table_name in tables.keys():
        print(f'Generating report for {table_name}')
        real_data = real_tables[table_name]['df']
        syn_data = cleaned_tables[table_name]
        domain_dict = real_tables[table_name]['domain']

        if configs['general']['test_data_dir'] is not None:
            test_data = test_tables[table_name]['df']
        else:
            test_data = None

        gen_single_report(
            real_data, 
            syn_data,
            domain_dict,
            table_name,
            save_dir,
            alpha_beta_sample_size=200_000,
            test_data=test_data
        )