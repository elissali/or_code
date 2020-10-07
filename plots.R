# modifying https://github.com/yuxingch/Implicature-Strength-Some/blob/master/analysis/rscripts/plots.R 

library(Hmisc)
library(gridExtra)
library(MuMIn)
library(tidyverse)
library(magrittr)

# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

source("helper.R")


# qualitative analysis (either, DE) 
preds = read_csv("./datasets/preds_bert_large_lstm_attn_original.csv")              # load the csv with model preds
raw = read_csv("./datasets/raw_data_full_3.csv")                                    # load the csv with annotations 
raw = raw %<>% 
    select(tgrep.id, entire.sentence, eitherPresent, DE)                            # make a tibble with only these columns
colnames(raw) = c("Item_ID","Sentence","Either","DE")
raw = unique(raw)                                                                   # remove duplicates
preds = preds %<>%                                                                  # make a tibble with only these columns
    select(Item_ID, predicted_m)
colnames(preds) = c("Item_ID","Preds")
dqual = merge(raw, preds, by="Item_ID")                                             # merge the annotation df with the preds df
dqual %<>%                                                                          # rename the column values
    mutate(Sentence=fct_reorder(Sentence,Preds)) %>%
    mutate_if(is.logical,funs(factor(.))) %>%
    mutate(Either = fct_recode(Either,"either \n present"="yes","either \n absent" ="no"),
           DE = fct_recode(DE,"downward \n entailing"="DE","upward \n entailing"="UE"))

dqual$Either = factor(dqual$Either, levels=c("either \n present", "either \n absent"), ordered = T) # save these as factor vecs
dqual$DE = factor(dqual$DE, levels=c("downward \n entailing", "upward \n entailing"), ordered=T)

dqual_all = dqual %>%                                           
  mutate(Predictor="Either", Value=Either)

dqual_all = rbind(dqual_all,                                                        # concat to get new dqual_all df (like an outer join)
                  dqual %>%
                  mutate(Predictor="DE", Value=DE))

eith_means = dqual %>%                                                              # get the means
  group_by(Either) %>%
  summarise(Prediction = mean(Preds),CILow=ci.low(Preds),CIHigh=ci.high(Preds)) %>%
  ungroup() %>%
  mutate(YMin=Prediction-CILow,YMax=Prediction+CIHigh) %>%
  mutate(Predictor="Either") %>% rename(Value=Either)

DE_means = dqual %>%
  group_by(DE) %>%
  summarise(Prediction = mean(Preds),CILow=ci.low(Preds),CIHigh=ci.high(Preds)) %>%
  ungroup() %>%
  mutate(YMin=Prediction-CILow,YMax=Prediction+CIHigh) %>%
  mutate(Predictor="DE") %>% rename(Value=DE)

means = rbind(eith_means, DE_means)                                                 # means is a 4x7 tibble 
means$Predictor = factor(means$Predictor, levels=c("Either", "DE"), ordered=T)

dodge = position_dodge(.9)
jitter = position_jitter(width = .2)

qualplot = ggplot(means, aes(x=Value,y=Prediction)) +                               # plot the means
  geom_point(data=dqual_all,aes(y=Preds),alpha=.3, color=cbPalette[2], position=jitter,size=2) +
  geom_errorbar(aes(ymin=YMin, ymax=YMax), width=.2, position=position_dodge(.9)) +
  geom_point(position=dodge,color="black", fill=cbPalette[6], pch=21,size=7) +
  scale_fill_manual(values=cbPalette[c(1,2)]) +
  scale_color_manual(values=cbPalette[c(1,2)]) + theme_bw() +
  theme(legend.position = "bottom", axis.title.x = element_blank(), axis.text.x = element_text(size=9), strip.text = element_text(size=12)) +
  facet_wrap(~Predictor, scales = "free_x")

ggsave("./graphs/qualitative_3.pdf", plot=qualplot, width=5,height=4)               # save the graph