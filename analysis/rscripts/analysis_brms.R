library(brms)
library(stringr)
library(tidyverse)
library(lme4)
library(lmerTest)
library(magrittr)



#setwd("analysis/rscripts/")

setwd("~/Dropbox/Uni/RA/implicature-strength/or/analysis/rscripts/")
source("./helpers.R")
theme_set(theme_bw(10))

d = read.csv("../data/raw_data_full_exp3.csv")
means = read.csv("../data/means_full_exp3.csv")

#### ANALYSIS
# visual exploration of feature effects on SI rating
cbPalette <- c("#56B4E9", "#D55E00", "#009E73","#999999", "#E69F00","#009E73","#56B4E9", "#D55E00", "#009E73","#999999", "#E69F00","#009E73","#56B4E9", "#D55E00", "#009E73","#999999", "#E69F00","#009E73","#56B4E9", "#D55E00", "#009E73","#999999", "#E69F00","#009E73")

predictors = c("cActually_test","cAnd_test", "Or_both_testchange","Or_both_testnotavailable", "clogSentenceLength","cNumeral","corNotPresent","credSentenceType","cresponse_goodsentence","cTrial","Disjunct_relationA~B","Disjunct_relationA>B","Intercept",   "cDE","ceitherPresent", "predicted_m")
predictor_labels = c("'actually' insertion test", "'and' replacement test", "'or both' test changed meaning", "'or both' test N/A",  "Log sentence length", "Numeral disjuncts", "'or not' present", "Sentence type", "Good sentence?", "Trial", "Same-level disjuncts", "Nested disjuncts", "Intercept", "Downward-entailing context", "'either' present", "NN predictions")

predictors = rev(predictors)
predictor_labels = rev(predictor_labels)



d %<>% 
  mutate_if(is.character,as.factor) 

d$Or_both_test = relevel(d$Or_both_test,"nochange")
contrasts(d$Or_both_test)
contrasts(d$Weird) = c(1,0)
contrasts(d$Disjunct_relation)

centered = cbind(d,myCenter(d[,c("response_goodsentence","redSentenceType", "DE", "Numeral", "And_test", "Actually_test", "eitherPresent", "orNotPresent", "logSentenceLength", "age", "Trial", "gender","Disjunct_individuation","Weird")])) # fix weird centering (NA values)

table(centered$And_test,centered$cAnd_test) # positive: no change
table(centered$Numeral,centered$cNumeral) # positive: numeral
table(centered$redSentenceType,centered$credSentenceType) # positive: not-declarative
table(centered$Actually_test,centered$cActually_test) # positive: no change
table(centered$Disjunct_individuation,centered$cDisjunct_individuation) # positive: same
table(centered$DE,centered$cDE) # positive: UE
table(centered$eitherPresent,centered$ceitherPresent) # positive: present
table(centered$orNotPresent,centered$corNotPresent) # positive: present

pred_data.exp3 = read.csv("../data/preds_bert_large_lstm_attn_discrete_cv_exp3.csv") %>% rename(tgrep.id=Item_ID) %>% select(tgrep.id, predicted_m)
centered %<>% merge(pred_data.exp3, by=c("tgrep.id"))


m.0 = lmer(response_val ~  (1|workerid), data=centered)
summary(m.0)
d$FittedOnlySubjectVar = fitted(m.0)

bm.0 = brm(response_val ~  (1|workerid), data=centered)
summary(bm.0)

means = d %>%
  group_by(tgrep.id) %>%
  dplyr::summarize(MeanPredicted = mean(FittedOnlySubjectVar),MeanEmpirical = mean(response_val))
ggplot(means, aes(x=MeanPredicted,y=MeanEmpirical)) +
  geom_point() +
  geom_smooth(method="lm") +
  xlim(0,1) +
  ylim(0,1) +
  ylab("Empirical rating") +
  xlab("Predicted rating")
ggsave("../graphs/model_fit_0.pdf",width=5,height=4)
cor(means$MeanEmpirical,means$MeanPredicted)

m.noitems = lmer(response_val ~  cDE + cAnd_test + Or_both_test + cNumeral + Disjunct_relation + ceitherPresent + corNotPresent + cActually_test + credSentenceType +  cresponse_goodsentence + clogSentenceLength + cTrial + (1|workerid), data=centered) # cDisjunct_individuation
summary(m.noitems)
d$FittedNoCluster = fitted(m.noitems)

bm.noitems = brm(response_val ~  cDE + cAnd_test + Or_both_test + cNumeral + Disjunct_relation + ceitherPresent + corNotPresent + cActually_test + credSentenceType +  cresponse_goodsentence + clogSentenceLength + cTrial + (1|workerid), data=centered) # cDisjunct_individuation
summary(bm.noitems)

means = d %>%
  group_by(tgrep.id) %>%
  dplyr::summarize(MeanPredicted = mean(FittedNoCluster),MeanEmpirical = mean(response_val))
ggplot(means, aes(x=MeanPredicted,y=MeanEmpirical)) +
  geom_point() +
  geom_smooth(method="lm") +
  xlim(0,1) +
  ylim(0,1) +
  ylab("Empirical rating") +
  xlab("Predicted rating")
ggsave("../graphs/model_fit_fixed.pdf",width=5,height=4)
cor(means$MeanEmpirical,means$MeanPredicted)
#r.squaredGLMM(m.noitems) 
#vif.mer(m.noitems) # no relevant collinearity

# random by-participant and by-item effects

#m.items = lmer(response_val ~  cDE + cAnd_test + Or_both_test + cNumeral + Disjunct_relation + ceitherPresent + corNotPresent + cActually_test + credSentenceType +  cresponse_goodsentence + clogSentenceLength + cTrial + (1|workerid) + (1|tgrep.id), data=centered) 
#summary(m.items)
#d$FittedNoClusterItem = fitted(m.items)
#
#means = d %>%
#  group_by(tgrep.id) %>%
#  dplyr::summarize(MeanPredicted = mean(FittedNoClusterItem),MeanEmpirical = mean(response_val))
#ggplot(means, aes(x=MeanPredicted,y=MeanEmpirical)) +
#  geom_point() +
#  geom_smooth(method="lm") +
#  xlim(0,1) +
#  ylim(0,1) +
#  ylab("Empirical rating") +
#  xlab("Predicted rating")
#ggsave("../graphs/model_fit_full.pdf",width=5,height=4)
#cor(means$MeanEmpirical,means$MeanPredicted)

# load prediction data


m.itemspred = lmer(response_val ~  predicted_m + cDE + cAnd_test + Or_both_test + cNumeral + Disjunct_relation + ceitherPresent + corNotPresent + cActually_test + credSentenceType +  cresponse_goodsentence + clogSentenceLength + cTrial + (1|workerid) + (1|tgrep.id), data=centered) 
summary(m.itemspred)

bm.noitemspred = brm(response_val ~  predicted_m + cDE + cAnd_test + Or_both_test + cNumeral + Disjunct_relation + ceitherPresent + corNotPresent + cActually_test + credSentenceType +  cresponse_goodsentence + clogSentenceLength + cTrial + (1|workerid), data=centered) 


bm.evalfixedandvectorpred.est = bm.noitemspred$fit %>% 
  as.data.frame() %>% 
  gather(key = "Parameter") %>% 
  filter(grepl("b_", Parameter)) %>% 
  group_by(Parameter) %>% 
  summarise(mu = mean(value), 
            cilow = quantile(value, 0.025), 
            cihigh = quantile(value, 0.975)) %>%
  mutate(model = "extended model")

bm.evalfixed.est = bm.noitems$fit %>% 
  as.data.frame() %>% 
  gather(key = "Parameter") %>% 
  filter(grepl("b_", Parameter)) %>% 
  group_by(Parameter) %>% 
  summarise(mu = mean(value), 
            cilow = quantile(value, 0.025), 
            cihigh = quantile(value, 0.975)) %>%
  mutate(model = "original model")



estimates = rbind(bm.evalfixedandvectorpred.est, bm.evalfixed.est)
estimates$Parameter = str_replace(estimates$Parameter, "b_", "")
estimates = estimates %>% filter(Parameter != "Intercept")
estimates$model = factor(estimates$model, levels = c("original model", "extended model"), ordered = TRUE)

max_values = estimates %>% group_by(Parameter) %>% summarise(mean_estimate = max(cihigh))

# simulated estimates of P(beta_fixed < beta_nn) for all fixed effects

bm.evalfixed_df = bm.noitems$fit %>% 
  as.data.frame()

params = colnames(bm.evalfixed_df)[grepl("b_", colnames(bm.evalfixed_df))]

bm.evalfixed_df = bm.evalfixed_df %>%
  select(params)

bm.evalfixedandvectorpred_df = bm.noitemspred$fit %>%
  as.data.frame() %>%
  select(params)


n = 1000000
fixed_sample = bm.evalfixed_df %>% sample_n(size = n, replace = T)
nn_sample = bm.evalfixedandvectorpred_df %>% sample_n(size = n, replace = T)
sim_results = data.frame(abs(fixed_sample) < abs(nn_sample))


sim_results %>% 
  gather(key="Parameter") %>% 
  group_by(Parameter) %>% 
  summarise("P(beta_fixed < beta_nn)" = mean(value))

signif_values = sim_results %>% 
  gather(key="Parameter") %>% 
  group_by(Parameter) %>% 
  summarise(p = mean(value)) %>%
  mutate(label=cut(p, breaks=c(-.01,0.001,0.01,0.05,1.0),
                   labels=c("***", "**", "*", ""))) %>%
  mutate(Parameter = factor(str_replace(str_replace_all(Parameter, "\\.", ":"), "b_", ""))) %>%
  merge(max_values)

estimates_plot = estimates %>% ggplot(aes(y=Parameter, x=mu)) + 
  geom_vline(xintercept=0) +
  geom_errorbarh(aes(xmin=cilow, xmax=cihigh, col=model), size=1, height=.4) +
  geom_point(aes(fill=model, pch=model), size=4, col="black") + 
  xlab("Coefficient estimate") +
  ylab("Parameter") +
  scale_shape_manual(name="Regression model", values=c(23,24)) +
  geom_text(aes(x=mean_estimate, label=label),col="black", size=5, data=signif_values, nudge_x=.1, nudge_y=-.1) + 
  theme(legend.position = "bottom")  +
  scale_color_manual(name="Regression model", values = cbPalette[c(2,1)]) +
  scale_fill_manual(name="Regression model", values = cbPalette[c(2,1)])




#############################


d = read.csv("../data/raw_data_full_exp4.csv")

#### ANALYSIS
# visual exploration of feature effects on SI rating
cbPalette <- c("#56B4E9", "#D55E00", "#009E73","#999999", "#E69F00","#009E73","#56B4E9", "#D55E00", "#009E73","#999999", "#E69F00","#009E73","#56B4E9", "#D55E00", "#009E73","#999999", "#E69F00","#009E73","#56B4E9", "#D55E00", "#009E73","#999999", "#E69F00","#009E73")



d %<>% 
  mutate_if(is.character,as.factor) 

d$Or_both_test = relevel(d$Or_both_test,"nochange")
contrasts(d$Or_both_test)
contrasts(d$Weird) = c(1,0)
contrasts(d$Disjunct_relation)

centered = cbind(d,myCenter(d[,c("response_goodsentence","redSentenceType", "DE", "Numeral", "And_test", "Actually_test", "eitherPresent", "orNotPresent", "logSentenceLength", "age", "Trial", "gender","Disjunct_individuation","Weird")])) # fix weird centering (NA values)

table(centered$And_test,centered$cAnd_test) # positive: no change
table(centered$Numeral,centered$cNumeral) # positive: numeral
table(centered$redSentenceType,centered$credSentenceType) # positive: not-declarative
table(centered$Actually_test,centered$cActually_test) # positive: no change
table(centered$Disjunct_individuation,centered$cDisjunct_individuation) # positive: same
table(centered$DE,centered$cDE) # positive: UE
table(centered$eitherPresent,centered$ceitherPresent) # positive: present
table(centered$orNotPresent,centered$corNotPresent) # positive: present

pred_data.exp4 = read.csv("../data/preds_bert_large_lstm_attn_discrete_cv_exp4.csv") %>% rename(tgrep.id=Item_ID) %>% select(tgrep.id, predicted_m)
centered %<>% merge(pred_data.exp4, by=c("tgrep.id"))


m.0 = lmer(response_val ~  (1|workerid), data=centered)
summary(m.0)
d$FittedOnlySubjectVar = fitted(m.0)

bm.0 = brm(response_val ~  (1|workerid), data=centered)
summary(bm.0)

means = d %>%
  group_by(tgrep.id) %>%
  dplyr::summarize(MeanPredicted = mean(FittedOnlySubjectVar),MeanEmpirical = mean(response_val))
ggplot(means, aes(x=MeanPredicted,y=MeanEmpirical)) +
  geom_point() +
  geom_smooth(method="lm") +
  xlim(1,7) +
  ylim(1,7) +
  ylab("Empirical rating") +
  xlab("Predicted rating")
ggsave("../graphs/model_fit_0.pdf",width=5,height=4)
cor(means$MeanEmpirical,means$MeanPredicted)

m.noitems = lmer(response_val ~  cDE + cAnd_test + Or_both_test + cNumeral + Disjunct_relation + ceitherPresent + corNotPresent + cActually_test + credSentenceType +  cresponse_goodsentence + clogSentenceLength + cTrial + (1|workerid), data=centered) # cDisjunct_individuation
summary(m.noitems)
d$FittedNoCluster = fitted(m.noitems)

bm.noitems = brm(response_val ~  cDE + cAnd_test + Or_both_test + cNumeral + Disjunct_relation + ceitherPresent + corNotPresent + cActually_test + credSentenceType +  cresponse_goodsentence + clogSentenceLength + cTrial + (1|workerid), data=centered) # cDisjunct_individuation
summary(bm.noitems)

means = d %>%
  group_by(tgrep.id) %>%
  dplyr::summarize(MeanPredicted = mean(FittedNoCluster),MeanEmpirical = mean(response_val))
ggplot(means, aes(x=MeanPredicted,y=MeanEmpirical)) +
  geom_point() +
  geom_smooth(method="lm") +
  xlim(0,1) +
  ylim(0,1) +
  ylab("Empirical rating") +
  xlab("Predicted rating")
ggsave("../graphs/model_fit_fixed.pdf",width=5,height=4)
cor(means$MeanEmpirical,means$MeanPredicted)
#r.squaredGLMM(m.noitems) 
#vif.mer(m.noitems) # no relevant collinearity

# random by-participant and by-item effects

#m.items = lmer(response_val ~  cDE + cAnd_test + Or_both_test + cNumeral + Disjunct_relation + ceitherPresent + corNotPresent + cActually_test + credSentenceType +  cresponse_goodsentence + clogSentenceLength + cTrial + (1|workerid) + (1|tgrep.id), data=centered) 
#summary(m.items)
#d$FittedNoClusterItem = fitted(m.items)
#
#means = d %>%
#  group_by(tgrep.id) %>%
#  dplyr::summarize(MeanPredicted = mean(FittedNoClusterItem),MeanEmpirical = mean(response_val))
#ggplot(means, aes(x=MeanPredicted,y=MeanEmpirical)) +
#  geom_point() +
#  geom_smooth(method="lm") +
#  xlim(0,1) +
#  ylim(0,1) +
#  ylab("Empirical rating") +
#  xlab("Predicted rating")
#ggsave("../graphs/model_fit_full.pdf",width=5,height=4)
#cor(means$MeanEmpirical,means$MeanPredicted)

# load prediction data


m.itemspred = lmer(response_val ~  predicted_m + cDE + cAnd_test + Or_both_test + cNumeral + Disjunct_relation + ceitherPresent + corNotPresent + cActually_test + credSentenceType +  cresponse_goodsentence + clogSentenceLength + cTrial + (1|workerid) + (1|tgrep.id), data=centered) 
summary(m.itemspred)

bm.noitemspred = brm(response_val ~  predicted_m + cDE + cAnd_test + Or_both_test + cNumeral + Disjunct_relation + ceitherPresent + corNotPresent + cActually_test + credSentenceType +  cresponse_goodsentence + clogSentenceLength + cTrial + (1|workerid), data=centered) 

bm.noitemspred = readRDS(file = "../rmodels/bm_noitemspred_exp4.rds")
bm.noitems = readRDS(file = "../rmodels/bm_noitems_exp4.rds")


bm.evalfixedandvectorpred.est = bm.noitemspred$fit %>% 
  as.data.frame() %>% 
  gather(key = "Parameter") %>% 
  filter(grepl("b_", Parameter)) %>% 
  group_by(Parameter) %>% 
  summarise(mu = mean(value), 
            cilow = quantile(value, 0.025), 
            cihigh = quantile(value, 0.975)) %>%
  mutate(model = "extended model")

bm.evalfixed.est = bm.noitems$fit %>% 
  as.data.frame() %>% 
  gather(key = "Parameter") %>% 
  filter(grepl("b_", Parameter)) %>% 
  group_by(Parameter) %>% 
  summarise(mu = mean(value), 
            cilow = quantile(value, 0.025), 
            cihigh = quantile(value, 0.975)) %>%
  mutate(model = "original model")



estimates = rbind(bm.evalfixedandvectorpred.est, bm.evalfixed.est)
estimates$Parameter = str_replace(estimates$Parameter, "b_", "")
estimates$Parameter = factor(estimates$Parameter, levels=predictors, labels=predictor_labels, ordered = TRUE)
estimates = estimates %>% filter(Parameter != "Intercept")
estimates$model = factor(estimates$model, levels = c("original model", "extended model"), ordered = TRUE)

max_values = estimates %>% group_by(Parameter) %>% summarise(mean_estimate = max(cihigh))

# simulated estimates of P(beta_fixed < beta_nn) for all fixed effects

bm.evalfixed_df = bm.noitems$fit %>% 
  as.data.frame()

params = colnames(bm.evalfixed_df)[grepl("b_", colnames(bm.evalfixed_df))]

bm.evalfixed_df = bm.evalfixed_df %>%
  select(params)

bm.evalfixedandvectorpred_df = bm.noitemspred$fit %>%
  as.data.frame() %>%
  select(params)


n = 1000000
fixed_sample = bm.evalfixed_df %>% sample_n(size = n, replace = T)
nn_sample = bm.evalfixedandvectorpred_df %>% sample_n(size = n, replace = T)
sim_results = data.frame(abs(fixed_sample) < abs(nn_sample))



sim_results %>% 
  gather(key="Parameter") %>% 
  group_by(Parameter) %>% 
  summarise("P(beta_fixed < beta_nn)" = mean(value))

signif_values = sim_results %>% 
  gather(key="Parameter") %>% 
  group_by(Parameter) %>% 
  summarise(p = mean(value)) %>%
  mutate(label=cut(p, breaks=c(-.01,0.001,0.01,0.05,1.0),
                   labels=c("***", "**", "*", ""))) %>%
  mutate(Parameter = factor(str_replace(str_replace_all(Parameter, "\\.", ":"), "b_", ""), levels=predictors, labels = predictor_labels, ordered = T)) %>%
  merge(max_values)

# reverse coding of DE
estimates[estimates$Parameter == "Downward-entailing context",]$mu = estimates[estimates$Parameter == "Downward-entailing context",]$mu * -1
estimates[estimates$Parameter == "Downward-entailing context",]$cilow = estimates[estimates$Parameter == "Downward-entailing context",]$cilow * -1
estimates[estimates$Parameter == "Downward-entailing context",]$cihigh = estimates[estimates$Parameter == "Downward-entailing context",]$cihigh * -1


estimates_plot = estimates %>% mutate(caption="Fig. 2: Regression coefficients for replication study") %>% ggplot(aes(y=Parameter, x=mu)) + 
  geom_vline(xintercept=0) +
  geom_errorbarh(aes(xmin=cilow, xmax=cihigh, col=model), size=1, height=.4) +
  geom_point(aes(fill=model, pch=model), size=4, col="black") + 
  xlab("Coefficient estimate") +
  ylab("Parameter") +
  scale_shape_manual(name="Regression model", values=c(23,24)) +
  geom_text(aes(x=mean_estimate, label=label),col="black", size=5, data=signif_values, nudge_x=.1, nudge_y=-.1) + 
  theme(legend.position = "bottom")  +
  scale_color_manual(name="Regression model", values = cbPalette[c(2,1)]) +
  scale_fill_manual(name="Regression model", values = cbPalette[c(2,1)]) +
  geom_rect(    aes(xmin = -0.6, xmax = 1.1, ymin = 0.55, ymax = 3.5),
    data = data.frame(),
    inherit.aes = FALSE,
    size = 0.75, lty=3, color =  "#666666", fill="transparent"
  ) 

ggsave(estimates_plot, filename = "../graphs/regression-coefficients_both.pdf", width=8, height=4)


# only original model
estimates_plot = estimates %>% filter(model == "original model") %>% ggplot(aes(y=Parameter, x=mu)) + 
  geom_vline(xintercept=0) +
  geom_errorbarh(aes(xmin=cilow, xmax=cihigh, col=model), size=1, height=.4) +
  geom_point(aes(fill=model, pch=model), size=4, col="black") + 
  xlab("Coefficient estimate") +
  ylab("Parameter") +
  scale_shape_manual(name="Regression model", values=c(23,24)) +
  scale_color_manual(name="Regression model", values = cbPalette[c(2,1)]) +
  scale_fill_manual(name="Regression model", values = cbPalette[c(2,1)]) +
  geom_rect(    aes(xmin = -0.6, xmax = 1.1, ymin = 0.55, ymax = 2.5),
                data = data.frame(),
                inherit.aes = FALSE,
                size = 0.75, lty=3, color =  "#666666", fill="transparent"
  )  +   theme(legend.position = "none")


estimates_plot
ggsave(estimates_plot, filename = "../graphs/regression-coefficients_original.pdf", width=8, height=4)


