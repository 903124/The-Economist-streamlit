"""
## APP NAME

DESCRIPTION

Author: [YOUR NAME](https://URL_TO_YOU))\n
Source: [Github](https://github.com/URL_TO_CODE)
"""
import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
from scipy import special
import plotly.express as px
import plotly.graph_objects as go

# Your imports goes below
rng = np.random.default_rng()
def draw_samples(mu, Sigma, ev,biden_states = None, trump_states = None, states = None, 
                         upper_biden = None, lower_biden = None, print_acceptance = False, target_nsim = 1000):
    sim = pd.DataFrame({})
    n = 0
    while(len(sim) < target_nsim):
        
        n += 1
        proposals = pd.DataFrame(scipy.special.expit(rng.multivariate_normal(mu, Sigma, int(1e5), method = "svd")),columns=list(ev.keys()))
        if(biden_states != None):
            for state in biden_states:
                proposals = proposals.where((proposals.loc[:,state] > 0.5).squeeze(),np.nan)
        
        if(trump_states != None):
            for state in trump_states:
                proposals = proposals.where((proposals.loc[:,state] < 0.5).squeeze(),np.nan)
        if(states != None):
            for i,state in enumerate(states):
                proposals = proposals.where(((proposals.loc[:,state] > lower_biden[i]).squeeze() & (proposals.loc[:,state] < upper_biden[i]).squeeze()),np.nan)

        reject = proposals[pd.isna(proposals)]
        sim = pd.concat([sim,proposals.dropna()])
        if (len(sim) < target_nsim and len(sim)/(len(proposals)*n) < 1-99.99/100):
            # print("More than 99.99% of the samples are rejected; you should relax some contraints.")
            break
    return({"matrix": sim, "acceptance_rate": len(sim)/(len(proposals)*n)})
def update_prob(mu, Sigma, ev, biden_states = None, trump_states = None, biden_scores_list = None, target_nsim = 1000, show_all_states = True):
    if(biden_scores_list != None):
        states = list(biden_scores_list.keys())
        lower_biden  = np.array(list(biden_scores_list.values()))[:,0]/100
        upper_biden  = np.array(list(biden_scores_list.values()))[:,1]/100
    else:
        states = None
        upper_biden = None
        lower_biden = None

    sim  = draw_samples(mu, Sigma, ev,biden_states = biden_states, trump_states = trump_states, states = states, 
                      upper_biden = upper_biden, lower_biden = lower_biden, 
                      target_nsim = target_nsim)

    if sim['acceptance_rate'] <= 1-99.99/100:
        return "low acceptance rate"

    ev_dist = np.dot(sim['matrix'] > 0.5,np.array(list(ev.values())) )
    state_win = np.mean(sim['matrix'] > 0.5,axis=0)
    p = np.mean(ev_dist >= 270)
    sd = np.sqrt(p*(1-p)/len(ev_dist))
    # if(show_all_states):
    #     print('Pr(biden wins) by state, in %:\n')
    #     print(pd.DataFrame(round(100*state_win),columns=['%']).T)
    #     print("--------\n")
    # print("Pr(biden wins the electoral college) = {:.0f}%\n[nsim = {}; se = {}%]".format(round(100*p),len(ev_dist),round(sd*100,1)))
    # if(show_all_states):
    #     print("--------\n")
    return state_win, p, sd, ev_dist


@st.cache
def read_file():
    sim_forecast = pd.read_csv('https://cdn.economistdatateam.com/us-2020-forecast/data/president/electoral_college_simulations.csv')
    sim_forecast = sim_forecast.loc[:, 'AK':]

    sim_forecast = sim_forecast + np.random.normal(0, 0.01, (len(sim_forecast),1))  # national error component
    sim_forecast = sim_forecast + np.random.normal(0, 0.02, (len(sim_forecast),len(sim_forecast.columns))) # state

    # this bit is really hacky
    # it make the simulations a little less correlated by add a bunch of random noise
    # this helps our live model react to really implausible events that could happen on election night
    # but will have the result of pushing the initial pre-results forecasts back toward 50-50

    sim_forecast = sim_forecast.clip(lower=0.0001,upper=0.99999)

    # now, get electoral votes in each state 
    # and make sure they're in the right order

    ev_state = pd.read_csv('https://raw.githubusercontent.com/TheEconomist/us-potus-model/master/data/2012.csv')[['ev','state']].sort_values('state')
    ev = ev_state.set_index('state').to_dict()['ev']

    sim_evs = ((sim_forecast> 0.5).astype(int) * np.array(list(ev.values()))).sum(axis=1)



    # adding ME1 and ME2, NE1 NE2 to sim_forecast matrix and ev vector
    # we do this by adding the average district-level two-party dem presidential vote, relative
    # to the state-level dem two-party vote, back to to our state-level forecast

    ev["ME"] = 2
    ev["NE"] = 2
    ev.update({"ME1":1, "ME2":1, "NE1":1, "NE2":1, "NE3":1})

    pres_results_by_cd = pd.read_csv('https://github.com/903124/political_data/raw/main/pres_results_by_cd.csv', index_col=0)
    me_ne_leans  = pres_results_by_cd[(pres_results_by_cd['year'] >= 2012) & (pres_results_by_cd['state_abb'].isin(['ME','NE']))]
    me_ne_leans = me_ne_leans.drop(['other'], axis=1).rename(columns={'state_abb': 'state'})
    me_ne_leans['sum_pct'] = me_ne_leans[['dem','rep']].sum(axis=1)
    me_ne_leans['total_votes'] = me_ne_leans['total_votes'] * me_ne_leans['sum_pct']
    me_ne_leans['dem'] = me_ne_leans['dem']/me_ne_leans['sum_pct']
    me_ne_leans['rep'] = me_ne_leans['rep']/me_ne_leans['sum_pct']
    me_ne_leans['dems_district_vote'] =  me_ne_leans['dem'] * me_ne_leans['total_votes']
    me_ne_leans['dem_vote_state'] =  (me_ne_leans.groupby(['state','year'])['dems_district_vote'].transform('sum') / me_ne_leans.groupby(['state','year'])['total_votes'].transform('sum'))
    me_ne_leans['dem_cd_lean'] = me_ne_leans['dem'] - me_ne_leans['dem_vote_state'] 
    me_ne_leans =(me_ne_leans[me_ne_leans.year == 2012].groupby(['state','district'])['dem_cd_lean'].sum()*0.3 + me_ne_leans[me_ne_leans.year == 2016].groupby(['state','district'])['dem_cd_lean'].sum()*0.7).reset_index()
    
    # create simulations for ME and NE districts


    # bind new simulation columns for the congressional districts, based on the above

    sim_forecast[['ME1','ME2']] =  (pd.concat([sim_forecast[['ME']]]*2,axis=1) + np.random.normal( me_ne_leans[me_ne_leans.state == 'ME']['dem_cd_lean'], 0.0075, (len(sim_forecast),len(me_ne_leans[me_ne_leans.state == 'ME']['dem_cd_lean'].index)))).values
    sim_forecast[['NE1','NE2','NE3']] =   (pd.concat([sim_forecast[['NE']]]*3,axis=1) + np.random.normal( me_ne_leans[me_ne_leans.state == 'NE']['dem_cd_lean'], 0.0075, (len(sim_forecast),len(me_ne_leans[me_ne_leans.state == 'NE']['dem_cd_lean'].index)))).values

    sim_forecast = sim_forecast.clip(lower=0.0001,upper=0.99999)
    sim_evs = ((sim_forecast> 0.5).astype(int) * np.array(list(ev.values()))).sum(axis=1)

    mu = np.mean(special.logit(sim_forecast),axis=0)
    Sigma = np.cov(special.logit(sim_forecast), rowvar=False)



    return mu, Sigma, ev




def main():
    st.header("The Economist election model simulation")
    st.write("The R version of the code is kindly provided by G. Elliot Morris from The Economist")
    st.write("https://gist.github.com/elliottmorris/c70fd4d32049c9986a45e2dfc07fb4f0\n")
    st.write('The code takes in The Economist election prediction model and allow user to do simulation base on true election outcome. The adjustmentable parameters are states won by candidate and lower/upper bound of vote share in the state.')


    mu, Sigma, ev = read_file()

    biden_states = []
    trump_states = []
    biden_share_list = {}

    col0, col1, col2, col3 = st.beta_columns(4)
    sort_states_keys = sorted(ev.keys())
    
    box_0, box_1, box_2, box_3 = [[[] for i in range(14)] for j in range(4)]
    slider_0, slider_1, slider_2, slider_3 = [[[] for i in range(14)] for j in range(4)]
    for i in range(14):

        box_0[i] = col0.selectbox(str(sort_states_keys[i]),('None','Trump','Biden'),key='box_'+str(sort_states_keys[i]))
        box_1[i] = col1.selectbox(str(sort_states_keys[14+i]),('None','Trump','Biden'),key='box_'+str(sort_states_keys[14+i]))
        box_2[i] = col2.selectbox(str(sort_states_keys[28+i]),('None','Trump','Biden'),key='box_'+str(sort_states_keys[28+i]))
        box_3[i] = col3.selectbox(str(sort_states_keys[42+i]),('None','Trump','Biden'),key='box_'+str(sort_states_keys[42+i]))
    
        slider_0[i] = col0.slider('Biden share%',min_value=0,max_value=100,value=(5, 95),key='slider_'+str(sort_states_keys[i]))
        slider_1[i] = col1.slider('Biden share%',min_value=0,max_value=100,value=(5, 95),key='slider_'+str(sort_states_keys[14+i]))
        slider_2[i] = col2.slider('Biden share%',min_value=0,max_value=100,value=(5, 95),key='slider_'+str(sort_states_keys[28+i]))
        slider_3[i] = col3.slider('Biden share%',min_value=0,max_value=100,value=(5, 95),key='slider_'+str(sort_states_keys[42+i]))


    for i, box_group in enumerate([box_0, box_1, box_2, box_3]):
        for j,box in enumerate(box_group):
            if box == 'Trump':
                trump_states.append(sort_states_keys[14*i+j])
            if box == 'Biden':
                biden_states.append(sort_states_keys[14*i+j])

    for i, slider_group in enumerate([slider_0, slider_1, slider_2, slider_3]):
        for j, slider in enumerate(slider_group):
            biden_share_list[sort_states_keys[14*i+j]] = [slider[0],slider[1]]
            
    with st.spinner('Sampling from simulation please wait...'):
        try:
            state_win, p, sd, ev_dist = update_prob(mu, Sigma, ev,biden_states = biden_states,trump_states = trump_states,biden_scores_list = None)

            st.write(pd.DataFrame({'Win %':round(100*state_win,1),'':''}).T)
            trump_win_chance = 100*len(ev_dist[ev_dist < 269])/float(len(ev_dist))
            st.write("Trump State: {}".format(trump_states))
            st.write("Biden State: {}".format(biden_states))
            st.write('')
            st.write("Trump win = {:.1f}%".format(trump_win_chance))
            layout = go.Layout(title = 'Simulation of electoral vote',xaxis = go.XAxis(title = 'Electoral Votes'),yaxis = go.YAxis(showticklabels=False))
            # fig = px.histogram(pd.DataFrame({'Electoral votes':ev_dist}), histnorm='probability density')
            fig = go.Figure(layout=layout)
            fig.add_trace(go.Histogram(x=ev_dist[ev_dist > 269],name='Biden win',xbins=dict(start=0,end=538,size=1),marker_color='#0000ff'))
            fig.add_trace(go.Histogram(x=ev_dist[ev_dist < 269],name='Trump win',xbins=dict(start=0,end=538,size=1),marker_color='#ff0000'))
            fig.add_trace(go.Histogram(x=ev_dist[ev_dist == 269],name='Draw',xbins=dict(start=0,end=538,size=1),marker_color='#bfbfbf'))
            # fig.update_traces(,marker_color='#FF0000')
            
            plot = st.plotly_chart(fig, use_container_width=True)

        except ValueError:
            st.warning('More than 99.99% of the samples are rejected; you should relax some contraints.')


if __name__ == "__main__":
    main()