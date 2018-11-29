
from IPython.display import display, Markdown
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from matplotlib import style
plt.style.use('fivethirtyeight')
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
sns.set_color_codes()
from os import listdir
from os.path import isfile, join

fontdict = {'fontsize':20, 'fontweight':'bold'}

"""
    Basic Structure should be like the below.

    ├── Project_Root
    │   ├── your_jupyter_notebook.ipynb
    │   ├── kaggle-survey-2018
    │   │   ├── SurveySchema.csv
    │   │   ├── freeFormResponses.csv
    │   │   └── multipleChoiceResponses.csv         

    You can download the data at 
    https://www.kaggle.com/kaggle/kaggle-survey-2018
"""

class KaggleSurvey:
    
    def __init__(self, is_update = False):
        self.__asset_path = "./"
        self.__dfs = {}
        self.__dict_2018 = {}
        print("Initializing.")
        files = [f for f in listdir(self.__asset_path) if isfile(join(self.__asset_path, f))] 
        if "schema_new.csv" not in files or is_update:
            self.__dict_2018 = self.__get_dfs()
            print("2018 files : ", list(self.__dict_2018.keys()))
            self.__save_dfs()
            self.__get_od_df()
            self.__make_q_list()
        else:
            self.__multi_18 = pd.read_csv(self.__asset_path + "multi_18.csv") 
            self.__free_18 = pd.read_csv(self.__asset_path + "free_18.csv") 
            self.__survey_schema_18 = pd.read_csv(self.__asset_path + "survey_schema_18.csv")
            self.__get_od_df()
            self.__make_q_list()
        print("Done.")
        
    def __get_dfs(self):
        path = self.__asset_path
        dirs = [f for f in listdir(path) if not isfile(join(path, f))]
        dict_2018 = {}
        for idx, dir_ in enumerate(dirs):
            df_path = path + dir_ + "/"
            files = [f for f in listdir(df_path) if isfile(join(df_path, f)) and "csv" in f]
            for idx_, file in enumerate(files):
                file_path = df_path + file
                df = pd.read_csv(file_path, encoding="ISO-8859-1")
                if idx > 0 and idx_ == 0:
                    display(Markdown("---"))
                display(Markdown(dir_.split("-")[-1] + " _**" + file + "**_ has <span style='color:blue'>"
                                 + str(df.shape) + "</span> shape of data."))
                dict_2018[file.split(".")[0]] = df

        return dict_2018
    
    def __save_dfs(self):
        q_list = self.__dict_2018["multipleChoiceResponses"].iloc[0].values.tolist()
        pd.concat(
            [pd.Series(self.__dict_2018["multipleChoiceResponses"].columns), pd.Series(q_list)], 1
        ).to_csv(self.__asset_path + 'schema_new.csv', index = False)

        self.__multi_18 = self.__dict_2018["multipleChoiceResponses"].iloc[1:].reset_index(drop=True)
        self.save_csv(self.__multi_18, "multi_18.csv")
        
        self.__free_18 = self.__dict_2018["freeFormResponses"].iloc[1:].reset_index(drop=True)
        self.save_csv(self.__free_18, "free_18.csv")
        
        cols = self.__dict_2018["SurveySchema"].columns
        rearrange_cols = ["Q" + str(i) for i in range(len(cols))]
        rearrange_cols[0], rearrange_cols[-1] = cols[0], cols[-1]
        self.__survey_schema_18 = self.__dict_2018["SurveySchema"][rearrange_cols]
        self.save_csv(self.__survey_schema_18, "survey_schema_18.csv")
    
    def __get_od_df(self):
        path = "./"
        files = [f for f in listdir(path) if isfile(join(path, f)) and "pkl" in f]
        if "od_df.pkl" in files:
            self.__od_df = pd.read_pickle("./od_df.pkl")
        else:
            self.__od_df = None
    
    def __make_q_list(self):
        single_choice, multiple_choice = self.__count_single_multiple_choices()
        q_list, is_single = self.__make_question_list(single_choice, multiple_choice)
        self.__q_df = pd.DataFrame({"question" : pd.Series(q_list), "is_single" : pd.Series(is_single)})
        self.__q_df.index = range(1, 51)
        # Question 12 is unique type of multiple choice.
        self.__q_df.at[12, 'is_single'] = 1
    
    def __count_single_multiple_choices(self):
        cols = self.__multi_18.columns
        single_choice = []
        multiple_choice = []
        for col in cols[1:]:
            tmp = col.split("_")
            if len(tmp) == 1:
                single_choice.append(col)
            elif "Part" in tmp:
                if tmp[0] not in multiple_choice:
                    multiple_choice.append(tmp[0])
        return single_choice, multiple_choice
        
    def __make_question_list(self, single_choice, multiple_choice):
        tmp_df = pd.read_csv(self.__asset_path + "/schema_new.csv")
        tmp_df.columns = ["Column", "Question"]
        tmp_df = tmp_df.set_index("Column")
        q_list = []
        q_len = len(single_choice) + len(multiple_choice)
        is_single_choice_list = []
        for i in range(q_len):
            is_single = 1
            q_txt = "Q" + str(i + 1)
            if q_txt in multiple_choice:
                q_txt = q_txt + "_Part_1"
                is_single = 0
            try:
                q = tmp_df.loc[q_txt]["Question"]
            except:
                q_txt = q_txt + "_TEXT"
                q = tmp_df.loc[q_txt]["Question"]
            q_list.append(q)
            is_single_choice_list.append(is_single)
        return q_list, is_single_choice_list

    def __get_selections_of_multiple_choice_question_as_list(self, number):
        tmp_df = pd.read_csv(self.__asset_path + "/schema_new.csv")
        tmp_df.columns = ["Column", "Question"]
        tmp_df = tmp_df.set_index("Column")
        tmp_li = [ 
            q_ for q_ in 
            (q for q in tmp_df.index.tolist() if str(number) in q) 
            if str(number) in q_.split("_")[0]
        ]
        return tmp_li
    
    def __print_question(self, year, col, cols=[], is_survey_schema=False, is_need_result = False):
        if is_survey_schema:
            tmp_df = pd.read_csv(self.__asset_path + "surveySchema_18.csv")
            print(tmp_df[col].iloc[0])
            return 
        dir_name = "kaggle-survey-"
        tmp_df = pd.DataFrame([])
        result = ""
        if year == 2017:
            tmp_df = pd.read_csv(self.__asset_path + dir_name + "2017/schema.csv")
        else:
            tmp_df = pd.read_csv(self.__asset_path + "/schema_new.csv")
            tmp_df.columns = ["Column", "Question"]
        tmp_df = tmp_df.set_index("Column")
        result = tmp_df.loc[col]["Question"]
        if is_need_result:
            return result
        else:
            print(col)
            if len(cols) == 0:
                print(result)
    
    def __save_order_df(self, question_number, new_order, is_rewrite_order):
        df = None
        if type(self.__od_df) == pd.core.frame.DataFrame:
            df = self.__od_df
            if str(question_number) not in df.question_number.values:
                df = df.append({str(question_number): new_order}, ignore_index=True)
            else:
                if is_rewrite_order:
                    idx = df[df.question_number == str(question_number)].index[0]
                    df.at[idx, "order"] = new_order
                
        else:
            df = pd.DataFrame({
                "question_number" : [str(question_number)],
                "order" : [new_order]
            })
        df.to_pickle("./od_df.pkl")
        
    def __get_order_li(self, df, q, question_number):
        if type(self.__od_df) == pd.core.frame.DataFrame:
            df_ = self.__od_df
            if str(question_number) not in df_.question_number.values:
                return [str_ for str_ in df[q].unique().tolist() if type(str_) != float]
            else:
                return df_[df_["question_number"] == str(question_number)].order.values[0]
        else:
            return [str_ for str_ in df[q].unique().tolist() if type(str_) != float]
    
    def __per_df(self, series) :
        val_cnt = series.values.sum()
        return series / val_cnt
    
    def __per_df_single(self, df, col, order_li):
        series = pd.Series(index = order_li)
        idx_li = df[col].value_counts().index.tolist()
        for idx in idx_li:
            series.at[idx] = df[col].value_counts().loc[idx]
        return series / series.sum()
    
    def get_question(self, number, is_need_display_order = False):
        """
            현재는 2018년도 문항들만 보여준다.
            number: 보고싶은 문제 번호를 입력
        """
        df = self.__q_df
        tmp = df.loc[number]
        if tmp["is_single"] == 1:
            q = tmp["question"]
            print("Q" +str(number) + ".", df.loc[number]["question"])
        else:
            tmp_li = self.__get_selections_of_multiple_choice_question_as_list(number)
            display_order = []
            for idx, q in enumerate(tmp_li):
                if "_OTHER_TEXT" not in q:                
                    q_ = self.__print_question(2018, q, is_need_result=True).split("-")
                    display_order.append(q_[-1].strip())
                    if idx == 0:
                        print(q.split("_")[0] + ".", q_[0])
                        if number == 35:
                            print(" ", str(idx + 1) + ".", " Self-" + q_[-1])
                        else:
                            print(" ", str(idx + 1) + ".", q_[-1])
                    else:
                        print(" ", str(idx + 1) + ".", q_[-1])
            if is_need_display_order:
                return display_order
    
    def get_o_df(self):
        return self.__od_df
    
    def get_q_df(self):
        """
            return df made of the question list
        """
        return self.__q_df
    
    def get_multiple_choice_responses(self):
        """
            return multipleChoiceResponses
        """
        return self.__multi_18
    
    def get_free_form_responses(self):
        """
            return freeFormResponses
        """
        return self.__free_18

    def get_survey_schema(self):
        """
            return SurveySchema
        """
        return self.__survey_schema_18
    
    def get_df_dictionary(self):
        """
            return the dictionary that includes all the dataset from kaggle survey 2018
            예:) 
                // 2018 files :  ['multipleChoiceResponses', 'freeFormResponses', 'SurveySchema']
                dict_2018["multipleChoiceResponses"].head()
        """

        self.__dict_2018 = self.__get_dfs() 
        print("2018 files : ", list(self.__dict_2018.keys()))
            
        return self.__dict_2018
    
    def get_df_saved_in_object(self):
        """
            return the dfs saved in object
        """
        return self.__dfs
        
    def get_df_I_want_by_country_name(self, country_name, year = 2018, which = "Country", is_false = False):
        """
            return certain country's the dataframe  
            country_name : insert the country full name
        """
        df = self.__multi_18.copy()
        col = ""
        if which == "Country" and year == 2018:
            col = "Q3"
        
        return df[df[col] != country_name] if is_false else df[df[col] == country_name]
    
    def set_df_I_want_by_country_name(self, countries = [], is_need_total = False):
        """
            (Recommended)
            You can save the datasets inside object. This way will make your notebook more clean.
        """
        
        keys = []
        values = []
        if is_need_total:
            keys.append("Total")
            values.append(self.__multi_18)
            
        countrie_names = self.__multi_18["Q3"].unique().tolist()
        for country in countries:
            if country not in countrie_names:
                print("Wrong country name. Country name should be the one of names below.")
                print(countrie_names)
                break
            df = self.get_df_I_want_by_country_name(country)
            keys.append(country)
            values.append(df)
                                
        self.__dfs = dict(zip(keys, values))

    def save_csv(self, df, filename, index=False):
        df.to_csv(self.__asset_path + filename, index = index)
        
    def draw_plot(self, question_number, plot_cols = 3, df = [], name = "Unnamed", dfs_ = {}, 
                  order = [], is_need_other = False, is_use_the_same_y_lim = True, ylim_offset = 0.1 ,is_rewrite_order = False):
        """
            question_number : int type of question number
            plot_cols : insert int type of how many cols you want
            df : Insert the dataframe you want to draw plow with
            name : If you insert only one dataframe to df parameter, you can set the title of plot. default is "Unnamed"
            dfs_ : Insert dictionary that includes dataframes
                example)
                    dfs = {
                        "US" : ks.get_df_I_want_by_country_name('United States of America'),
                        "India" : ks.get_df_I_want_by_country_name('India'),
                        "China" : ks.get_df_I_want_by_country_name('China')
                    }
            order : Insert the order of xlabels.
        """
        
        if len(df) != 0 and len(dfs_) != 0:
            print("You should insert either one dataframe or dictionary of dataframes.")
            print("Try again.")
            return
        
        dfs_keys = []
        dfs = []
        if len(df) != 0:
            dfs_keys = [name]
            dfs = [df]
        elif len(dfs_) != 0:
            dfs_keys = list(dfs_.keys())
            dfs = dfs_.values()
        elif len(self.__dfs) != 0:
            dfs_keys = list(self.__dfs.keys())
            dfs = self.__dfs.values()
        else:
            print("There was no Valid DataFrame.")
        
        is_single_choice_question = self.__q_df.loc[question_number]["is_single"]
        
        if is_single_choice_question:
            self.get_question(question_number)
            if question_number == 3:
                len_li = [len(df) for df in dfs]
                plt.bar(dfs_keys, len_li)
                
            else:        
                q = "Q" + str(question_number) + "_MULTIPLE_CHOICE" if question_number == 12 else "Q" + str(question_number)
                length_of_dfs = len(dfs)
                depth_of_dfs = int(np.ceil(length_of_dfs / float(plot_cols)))                    
                f, ax = plt.subplots(depth_of_dfs, plot_cols, figsize=(5 * length_of_dfs, 5 * depth_of_dfs))
                
                if is_use_the_same_y_lim:
                    max_value = 0
                    for df in dfs:
                        ncount = len(df)
                        tmp = df.groupby(q).size().values/ncount
                        tmp_max_value = (tmp.max() + ylim_offset)
                        max_value = tmp_max_value if tmp_max_value > max_value else max_value
                
                order_li = []
                print("Answers from left to right : ")
                
                for idx, df in enumerate(dfs):
                    
                    if idx == 0:
                        if len(order) != 0:
                            order_li = order
                            self.__save_order_df(question_number, order, is_rewrite_order)
                        else:
                            order_li = self.__get_order_li(df, q, question_number)
#                             order_li = [str_ for str_ in df[q].unique().tolist() if type(str_) != float]
                    ax_ = None
                    if plot_cols == 1:
                        ax_ = ax[idx] 
                    elif plot_cols > 1:
                        ax_ = ax[idx // plot_cols, idx % plot_cols] if depth_of_dfs > 1 else ax[idx % plot_cols]
                    else:
                        print("wrong plot_cols.")
                        return
#                     ax_ = ax[idx // plot_cols, idx % plot_cols] if depth_of_dfs > 1 else ax[idx % plot_cols]
    
                    self.__per_df_single(df, q, order_li).plot.bar(ax = ax_)
                    for p in ax_.patches:
                        ax_.annotate(str(round(p.get_height() * 100, 2)) + "%", (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
                    ax_.set_xticklabels(list(range(1, len(order_li) + 1)), rotation=0)
                    ax_.set_title(dfs_keys[idx], fontdict = fontdict)
                    if is_use_the_same_y_lim:
                        ax_.set_ylim(0, max_value)
                        
                for idx, answer in enumerate(order_li):
                    print(idx + 1, answer)
                plt.show()
                
        else:
            answers = self.get_question(question_number, is_need_display_order=True)
            cols = [
                q for q in self.__get_selections_of_multiple_choice_question_as_list(question_number) if "_OTHER_TEXT" not in q
            ] 
            order_li = []
            result_dfs = []
            length_of_dfs = len(dfs)
            depth_of_dfs = int(np.ceil(length_of_dfs / float(plot_cols)))
#             if depth_of_dfs > 1:
#                 display(Markdown("##### Plot row가 2개 이상이므로 Xlabel은 지문 내용을 참고하세요."))
            f, ax = plt.subplots(depth_of_dfs, plot_cols, figsize=(5 * length_of_dfs, 5 * depth_of_dfs))
            # 34, 35 answers must add up to 100%
            if question_number == 34 or question_number == 35:
                if is_use_the_same_y_lim:
                    max_value = 0
                    for df in dfs:
                        for col in cols:
                            mean_max = df[cols].mean().max()
                            tmp_max_value = (mean_max + ylim_offset * 100)
                            max_value = tmp_max_value if tmp_max_value > max_value else max_value
                
                for idx, df in enumerate(dfs):
                    ax_ = None
                    if plot_cols == 1:
                        ax_ = ax[idx] 
                    elif plot_cols > 1:
                        ax_ = ax[idx // plot_cols, idx % plot_cols] if depth_of_dfs > 1 else ax[idx % plot_cols]
                    else:
                        print("wrong plot_cols.")
                        return
#                     ax_ = ax[idx // plot_cols, idx % plot_cols] if depth_of_dfs > 1 else ax[idx % plot_cols]
                    sns.barplot(data = df[cols], ax = ax_)
                    ax_.set_xticklabels(list(range(1, len(df.columns) + 1)), rotation = 0)
                    ax_.set_title(dfs_keys[idx], fontdict = fontdict)
                    if is_use_the_same_y_lim:
                        ax_.set_ylim(0, max_value)
            else:
                for idx, df in enumerate(dfs):
                    x_li = []
                    y_li = []
                    for i, col in enumerate(cols):
                        uq_li = df[col].unique().tolist()
                        if len(uq_li) > 1:
                            if str(uq_li[0]) == "nan":
                                y_li.append(uq_li[1])
                                x_li.append(len(df[df[col] == uq_li[1]]))
                            else:
                                y_li.append(uq_li[0])
                                x_li.append(len(df[df[col] == uq_li[0]]))
                        else:
                            y_li.append(answers[i])
                            x_li.append(0)
                    result_dfs.append(pd.Series(x_li, y_li))
                    if len(order_li) == 0:
                        order_li = y_li

                if is_use_the_same_y_lim:
                    max_value = 0
                    for df in result_dfs:
                        tmp = df/df.sum()
                        tmp_max_value = (tmp.max() + ylim_offset)
                        max_value = tmp_max_value if tmp_max_value > max_value else max_value

                for idx, df in enumerate(result_dfs):
                    if idx == 0:
                        if len(order) != 0:
                            order_li = order
                    ax_ = None
                    if plot_cols == 1:
                        ax_ = ax[idx] 
                    elif plot_cols > 1:
                        ax_ = ax[idx // plot_cols, idx % plot_cols] if depth_of_dfs > 1 else ax[idx % plot_cols]
                    else:
                        print("wrong plot_cols.")
                        return
                    
                    self.__per_df(df).plot.bar(ax = ax_)
                    for p in ax_.patches:
                        ax_.annotate(str(round(p.get_height() * 100, 2)) + "%", (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
                    ax_.set_xticklabels(list(range(1, len(order_li) + 1)), rotation = 0)
                    ax_.set_title(dfs_keys[idx], fontdict = fontdict)
                    if is_use_the_same_y_lim:
                        ax_.set_ylim(0, max_value)
                plt.show()