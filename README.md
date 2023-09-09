{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "5e2bf554-3ee9-454d-81ab-6a7747a7f443",
   "metadata": {},
   "source": [
    "# Titanic Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "38a2f268-dc96-4e6b-8d2e-faeaf882d748",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import Libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cd7b2f9b-9482-47da-9927-216c3ccb892f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "      <th>embarked</th>\n",
       "      <th>class</th>\n",
       "      <th>who</th>\n",
       "      <th>adult_male</th>\n",
       "      <th>deck</th>\n",
       "      <th>embark_town</th>\n",
       "      <th>alive</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>First</td>\n",
       "      <td>woman</td>\n",
       "      <td>False</td>\n",
       "      <td>C</td>\n",
       "      <td>Cherbourg</td>\n",
       "      <td>yes</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>woman</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>yes</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>First</td>\n",
       "      <td>woman</td>\n",
       "      <td>False</td>\n",
       "      <td>C</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>yes</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.4583</td>\n",
       "      <td>Q</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Queenstown</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>51.8625</td>\n",
       "      <td>S</td>\n",
       "      <td>First</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>E</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>21.0750</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>child</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>11.1333</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>woman</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>yes</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>female</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0708</td>\n",
       "      <td>C</td>\n",
       "      <td>Second</td>\n",
       "      <td>child</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Cherbourg</td>\n",
       "      <td>yes</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   survived  pclass     sex   age  sibsp  parch     fare embarked   class  \\\n",
       "0         0       3    male  22.0      1      0   7.2500        S   Third   \n",
       "1         1       1  female  38.0      1      0  71.2833        C   First   \n",
       "2         1       3  female  26.0      0      0   7.9250        S   Third   \n",
       "3         1       1  female  35.0      1      0  53.1000        S   First   \n",
       "4         0       3    male  35.0      0      0   8.0500        S   Third   \n",
       "5         0       3    male   NaN      0      0   8.4583        Q   Third   \n",
       "6         0       1    male  54.0      0      0  51.8625        S   First   \n",
       "7         0       3    male   2.0      3      1  21.0750        S   Third   \n",
       "8         1       3  female  27.0      0      2  11.1333        S   Third   \n",
       "9         1       2  female  14.0      1      0  30.0708        C  Second   \n",
       "\n",
       "     who  adult_male deck  embark_town alive  alone  \n",
       "0    man        True  NaN  Southampton    no  False  \n",
       "1  woman       False    C    Cherbourg   yes  False  \n",
       "2  woman       False  NaN  Southampton   yes   True  \n",
       "3  woman       False    C  Southampton   yes  False  \n",
       "4    man        True  NaN  Southampton    no   True  \n",
       "5    man        True  NaN   Queenstown    no   True  \n",
       "6    man        True    E  Southampton    no   True  \n",
       "7  child       False  NaN  Southampton    no  False  \n",
       "8  woman       False  NaN  Southampton   yes  False  \n",
       "9  child       False  NaN    Cherbourg   yes  False  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Load the data\n",
    "titanic = sns.load_dataset('titanic')\n",
    "#Print the first 10 rows of data\n",
    "titanic.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "88f37b46-a647-4638-8c6b-375ef0d120d4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 15)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Count the number of rows and columns in the data set \n",
    "titanic.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fd8296c8-1c3c-4575-bfb5-3856a3fd7185",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         survived      pclass         age       sibsp       parch        fare\n",
       "count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000\n",
       "mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208\n",
       "std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429\n",
       "min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000\n",
       "25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400\n",
       "50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200\n",
       "75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000\n",
       "max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a931b804-0ba9-445a-8ef7-000af0685b75",
   "metadata": {},
   "source": [
    "Get a count of the number of survivors on board the Titanic in this data set. Notice that, in this data set, there were more passengers that didn’t survive (549) than did (343)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c1bd7ecc-a5e9-4242-9ce2-ef21a2eefe04",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    549\n",
       "1    342\n",
       "Name: survived, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Get a count of the number of survivors  \n",
    "titanic['survived'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "53be55a7-c924-43ab-ac1f-c18631949555",
   "metadata": {},
   "source": [
    "Visualize the number of survivors on board the Titanic in this data set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8a2c06c9-93b6-418a-bd8e-d8141c015101",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: ylabel='count'>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Visualize the count of number of survivors\n",
    "sns.countplot(titanic['survived'],label=\"Count\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50c2d2dc-8f42-4328-9a74-bff57db80340",
   "metadata": {},
   "source": [
    "Visualize the count of survivors for the columns who, sex, pclass, sibsp, parch, and embarked."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "11c70d5e-1aa4-418e-8feb-38a58edb0580",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sex</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>female</th>\n",
       "      <td>0.742038</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>male</th>\n",
       "      <td>0.188908</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        survived\n",
       "sex             \n",
       "female  0.742038\n",
       "male    0.188908"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Look at survival rate by sex\n",
    "titanic.groupby('sex')[['survived']].mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8cac12c-d754-4e9d-970a-dfc12c25f042",
   "metadata": {},
   "source": [
    "Males in third class had the lowest survival rate at about 13.54%, meaning the majority of them did not survive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ac38a5a6-670e-415a-bb29-4e959ca1bd48",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>class</th>\n",
       "      <th>First</th>\n",
       "      <th>Second</th>\n",
       "      <th>Third</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sex</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>female</th>\n",
       "      <td>0.968085</td>\n",
       "      <td>0.921053</td>\n",
       "      <td>0.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>male</th>\n",
       "      <td>0.368852</td>\n",
       "      <td>0.157407</td>\n",
       "      <td>0.135447</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "class      First    Second     Third\n",
       "sex                                 \n",
       "female  0.968085  0.921053  0.500000\n",
       "male    0.368852  0.157407  0.135447"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Look at survival rate by sex and class\n",
    "titanic.pivot_table('survived', index='sex', columns='class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "70038430-4e88-4799-801f-36652cbe5227",
   "metadata": {},
   "source": [
    "Let’s visualize the survival rate by sex and class."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "2fd81716-5792-4b45-889f-8f6fb75dfbf9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: xlabel='sex'>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Look at survival rate by sex and class visually\n",
    "titanic.pivot_table('survived', index='sex', columns='class').plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a1c3eaed-90b9-4bb7-90a8-52c1df106332",
   "metadata": {},
   "source": [
    "Visualize the survival rate by class using a bar plot.\n",
    "\n",
    "A little over 60% of the passengers in first class survived. Less than 30% of passengers in third class survived. That means less than half of the passengers in third class survived, compared to the passengers in first class."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "2486aecf-6efb-48f6-9222-a3ef647a3ea2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: xlabel='class', ylabel='survived'>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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evVSnz599OCDD7bpacQwy3f/LwwJCZHD4WjRdtLnn3+u0aNHq1u3brr66qs9ZiK/f1rqkUce0TXXXKNVq1bp0ksvldPplGVZ+vTTTzVq1CgFBgZq8ODBLe7Kj7Yh3MArTz/9tGJjY1VSUqKHHnpIS5Ys0caNG/Xaa6/pk08+0UsvveQOMe+//74kafXq1XK5XO51nJ969OihHj16aMOGDaqvr2/xvmVZuvHGG1VZWamCggKVlJRo6NChGjNmjHtm580339TEiRN14403qrS0VJs3b1Z8fLx7G9OmTdOOHTu0ceNGbd++XZZlacKECTp+/Li7z9dff61nnnlGL774ot59912Vl5d73O18wYIF7rugb926VUeOHNH69evP4k8GnV1WVpbmzp2rsrIyXX755Zo0aZIaGxtP2f+zzz7Ta6+9pnXr1qmsrEzNzc2aOHGi/Pz89N5772nZsmX69a9/fQ73wCAW8G933nmn5efnZ3Xv3t29/PKXv7Suu+46a/bs2e5+kZGRVnJyssfYe++917r++uut5ubmVrctyVq/fv1ZrB6dyRtvvGFdcMEFVmBgoJWYmGhlZmZaH374oWVZlrV582YrODjY+vbbbz3GDBw40Fq+fLllWZaVkJBg3XHHHa1ue/fu3ZYka9u2be626upqKygoyHrttdcsy7Ks1atXW5Kszz77zN3n2WeftcLCwtzr4eHh1hNPPOFeP378uNWvXz/rpptuat/Oo9NavXq1FRIS0qJ97969liTr+eefd7f985//tCRZu3btanXsww8/bHXt2tWqqqpyt23atMny8/OzKioq3G1vvfUW/396gZkbeBg9erTKysrcy5IlS1rt992/kqUTfymXlZUpOjpa9913n95+++1zUS46qVtuuUUHDx7Uxo0bNW7cOG3ZskVDhw7VmjVrVFJSoi+//FJ9+vRxz/L06NFDe/fu1Z49eyRJZWVlGjNmTKvb3rVrl/z9/TVs2DB3W58+fRQdHa1du3a527p166aBAwe618PDw1VVVSXpxGkxl8ulhIQE9/v+/v4tjnvgu6666ir36/DwcElyH1OtiYyMVN++fd3ru3btUv/+/dWvXz9323ePQbQdFxTDQ/fu3TVo0KA29fuuoUOHau/evXrrrbf0v//7v7rttts0duxYvfHGG2erVHRygYGBuuGGG3TDDTdo/vz5Sk1N1cMPP6z09HSFh4dry5YtLcacvGbhdBdmWqe4LsayLDkcDvf69y/edDgcXFODdvnuMXXyWGtubj5l/+//P9ra8ffdYxZtx8wNfCY4OFgpKSl67rnnlJ+fr3Xr1rmvkejatauamppsrhAd2eDBg/XVV19p6NChqqyslL+/vwYNGuSxhIaGSjrxF/LmzZtPuZ3Gxkb93//9n7utpqZGu3fvVkxMTJtqCQkJUXh4uN577z13W2Njo0pKStqxh8DpDR48WOXl5Tp48KC7jdsjeIeZG/jEokWLFB4ermuuuUZdunTR66+/rosuusj9l/aAAQO0efNmjRgxQk6nUxdccIG9BcM2NTU1uvXWWzV9+nRdddVV6tmzp3bs2KGnnnpKN910k8aOHauEhAQlJyfrySefVHR0tA4ePKiCggIlJycrPj5eDz/8sMaMGaOBAwfq9ttvV2Njo9566y09+OCDuuyyy3TTTTfprrvu0vLly9WzZ0/NmzdPl1xyiW666aY21zl79mw98cQTuuyyyxQTE6OFCxfq6NGjZ+8Hg/Pe2LFjFR0dralTp2rBggWqq6tTVlaW3WV1SszcwCd69OihJ598UvHx8frxj3+sffv2qaCgQF26nDjEFixYoMLCQkVERGjIkCE2Vws79ejRQ8OGDdOiRYs0atQoxcbG6qGHHtJdd92lP/zhD3I4HCooKNCoUaM0ffp0XX755br99tu1b98+hYWFSTpx1+zXX39dGzdu1DXXXKPrr7/eY6Zm9erViouL089+9jMlJCTIsiwVFBSc0X1E/uu//ktTp07VtGnTlJCQoJ49e+rmm2/2+c8DOKlLly5av3696uvrde211yo1NVW//e1v7S6rU3JYnGQGAAAGYeYGAAAYhXADAACMQrgBAABGIdwAAACjEG4AAIBRCDcAAMAohBsAAGAUwg0AADAK4QZAp7Fv3z45HA6VlZXZXQqADoxwAwAAjEK4AQAARiHcAOhwmpub9eSTT2rQoEFyOp3q379/qw8QbGpq0owZMxQVFaWgoCBFR0dr8eLFHn22bNmia6+9Vt27d1evXr00YsQI7d+/X5L04YcfavTo0erZs6eCg4MVFxenHTt2nJN9BHD2+NtdAAB8X2Zmpp577jktWrRIP/nJT+RyufTxxx+36Nfc3Kx+/frptddeU2hoqIqLi3X33XcrPDxct912mxobG5WcnKy77rpLr776qhoaGvT3v/9dDodDknTHHXdoyJAhWrp0qfz8/FRWVnZGTw4H0DHxVHAAHcqxY8fUt29f/eEPf1BqaqrHe/v27VNUVJRKS0t1zTXXtDr+nnvu0aFDh/TGG2/oyJEj6tOnj7Zs2aLrrruuRd/g4GD9/ve/15133nk2dgWATTgtBaBD2bVrl+rr6zVmzJg29V+2bJni4+PVt29f9ejRQ88995zKy8slSb1799a0adM0btw4/fznP9fixYvlcrncYzMyMpSamqqxY8fqiSee0J49e87KPgE4twg3ADqUoKCgNvd97bXXdP/992v69Ol6++23VVZWpl/96ldqaGhw91m9erW2b9+uxMRE5efn6/LLL9d7770nSXrkkUf0z3/+UzfeeKP++te/avDgwVq/fr3P9wnAucVpKQAdyrfffqvevXtryZIlP3ha6t5779XOnTu1efNmd5+xY8equrr6lPfCSUhI0I9//GMtWbKkxXuTJk3SV199pY0bN/p0nwCcW8zcAOhQAgMD9etf/1oPPvig1q5dqz179ui9997TypUrW/QdNGiQduzYoU2bNmn37t166KGH9P7777vf37t3rzIzM7V9+3bt379fb7/9tnbv3q2YmBh98803mjVrlrZs2aL9+/dr27Ztev/99xUTE3MudxfAWcC3pQB0OA899JD8/f01f/58HTx4UOHh4UpLS2vRLy0tTWVlZUpJSZHD4dCkSZOUnp6ut956S5LUrVs3ffzxx3rhhRdUU1Oj8PBwzZo1SzNnzlRjY6Nqamo0depUHTp0SKGhoZo4caIeffTRc727AHyM01IAAMAonJYCAABGIdwAAACjEG4AAIBRCDcAAMAohBsAAGAUwg0AADAK4QYAABiFcAMAAIxCuAEAAEYh3AAAAKMQbgAAgFH+Hx2+btFOClxuAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Plot the survival rate of each class.\n",
    "sns.barplot(x='class', y='survived', data=titanic)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0c7c10a7-cc34-4aa3-91b6-4d7f2d5b3988",
   "metadata": {},
   "source": [
    "Take a look at the survival rate by sex, age, and class.\n",
    "\n",
    "Note that, in this data set, the oldest person is aged 80, so that will be our age limit.\n",
    "\n",
    "We can see from the table below that women in first class that were 18 and older had the highest survival rate at 97.2973%, while men 18 and older in second class had the lowest survival rate of 7.1429%."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "4e337f4c-9af4-4726-9b13-2d6fa38d5c5f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>class</th>\n",
       "      <th>First</th>\n",
       "      <th>Second</th>\n",
       "      <th>Third</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">female</th>\n",
       "      <th>(0, 18]</th>\n",
       "      <td>0.909091</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.511628</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>(18, 80]</th>\n",
       "      <td>0.972973</td>\n",
       "      <td>0.900000</td>\n",
       "      <td>0.423729</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">male</th>\n",
       "      <th>(0, 18]</th>\n",
       "      <td>0.800000</td>\n",
       "      <td>0.600000</td>\n",
       "      <td>0.215686</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>(18, 80]</th>\n",
       "      <td>0.375000</td>\n",
       "      <td>0.071429</td>\n",
       "      <td>0.133663</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "class               First    Second     Third\n",
       "sex    age                                   \n",
       "female (0, 18]   0.909091  1.000000  0.511628\n",
       "       (18, 80]  0.972973  0.900000  0.423729\n",
       "male   (0, 18]   0.800000  0.600000  0.215686\n",
       "       (18, 80]  0.375000  0.071429  0.133663"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Look at survival rate by sex, age and class\n",
    "age = pd.cut(titanic['age'], [0, 18, 80])\n",
    "titanic.pivot_table('survived', ['sex', age], 'class')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1f36e730-eae5-4fb7-9361-8d539ebdcbe5",
   "metadata": {},
   "source": [
    "Plot the prices paid for each class."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "9b2d007b-5ae3-44e6-a7a7-754f27c3bb4a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAlQAAAHFCAYAAAA0SmdSAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAABCBUlEQVR4nO3deXgUVd728buz7xsEkhCSsMQIyiIGNIyCyL6MQURAGCCDMo4LwovIgMgmKigIis+APirgMg6CYIwoCKggM4DIpigIyBaUsMiSBAJZz/sHT1pasleHEPx+rqsv7apT1b86adJ3TlWdthljjAAAAFBhLlVdAAAAQHVHoAIAALCIQAUAAGARgQoAAMAiAhUAAIBFBCoAAACLCFQAAAAWEagAAAAsIlABAABYRKACIElasGCBbDab/eHm5qbIyEj99a9/1S+//FKmfSQlJSkmJqZyCy3Grl27lJSUpKioKHl4eKhmzZrq1q2bli9fXmT7zz//XPHx8fL19ZXNZlNycnKR7Q4ePOjQL79/TJo0qVKOp/DnsXnzZkv7Wbdunfr06aM6derIw8NDgYGBat26tebOnatz587Z28XExCgpKcli1cAfl1tVFwDg6jJ//nxdf/31On/+vL766itNnTpVa9eu1Y4dO+Tr61vituPHj9fw4cOvUKW/Wbp0qfr376/69etr/PjxiouL07FjxzR//nx169ZNTzzxhF544QV7e2OM+vTpo+uuu04pKSny9fVVXFxcia8xbNgw9e/f/7LlkZGRTj8eZ5k4caKefvpptW7dWlOmTFGDBg2UlZWl9evXa9KkSdqzZ49mzZpV1WUC1wQCFQAHN954o+Lj4yVJ7dq1U35+vqZMmaLk5GQNGDCgyG2ysrLk4+OjBg0aXMlSJUn79u3TwIED1aRJE61Zs8Yh9N1777166KGHNH36dLVo0UL9+vWTJB05ckSnTp3S3Xffrfbt25fpdaKionTrrbdWyjFUhsWLF+vpp5/W/fffr9dff102m82+rmvXrho9erQ2bNhQhRUC1xZO+QEoUWGIOHTokKSLp/X8/Py0Y8cOderUSf7+/vZQUtQpv4KCAr3yyitq3ry5vL29FRQUpFtvvVUpKSkO7d5//30lJCTI19dXfn5+6ty5s7Zt21ZqfbNmzVJWVpZeeeWVIkfQXnzxRQUFBenZZ5+VJE2aNMk+qvSPf/xDNpvNaacpV61apcTEREVGRsrLy0sNGzbUgw8+qF9//fWytj/++KPuu+8+1a5dW56enoqKitKgQYOUnZ3t0C4zM1MPPfSQatasqRo1aqhXr146cuRIqbU8/fTTCg4O1uzZsx3CVCF/f3916tSp2O0vXLigxx9/XM2bN1dgYKBCQkKUkJCgjz766LK2ixcv1i233KLAwED5+Piofv36GjJkiH19QUGBnnnmGcXFxdnfA02bNtXLL79c6nEA1QUjVABK9NNPP0mSQkND7ctycnJ011136cEHH9SYMWOUl5dX7PZJSUl69913df/99+vpp5+Wh4eHtm7dqoMHD9rbPPfcc3rqqaf017/+VU899ZRycnI0ffp03X777dq0aZMaN25c7P5XrVql2rVrFzt65OPjo06dOmnRokU6evSoHnjgATVr1ky9evWyn8bz9PQstR8KCgqKPE43t99+je7bt08JCQl64IEHFBgYqIMHD2rmzJm67bbbtGPHDrm7u0uSvv32W912222qWbOmnn76acXGxiotLU0pKSnKyclxqOeBBx5Q9+7d9d577+nw4cN64okn9Je//EVffPFFsbWmpaXp+++/V9++feXj41PqsRUlOztbp06d0qhRo1SnTh3l5ORo9erV6tWrl+bPn69BgwZJkjZs2KC+ffuqb9++mjRpkry8vHTo0CGH+l544QVNmjRJTz31lNq0aaPc3Fz9+OOPOnPmTIVqA65KBgCMMfPnzzeSzMaNG01ubq7JzMw0y5YtM6Ghocbf398cPXrUGGPM4MGDjSQzb968y/YxePBgEx0dbX/+1VdfGUlm3Lhxxb5uamqqcXNzM8OGDXNYnpmZacLCwkyfPn1KrNvLy8vceuutJbb5xz/+YSSZr7/+2hhjzIEDB4wkM3369BK3u7RtcY9169YVuV1BQYHJzc01hw4dMpLMRx99ZF935513mqCgIHP8+PFiX7fw5/Hwww87LH/hhReMJJOWllbsths3bjSSzJgxY0o9vkLR0dFm8ODBxa7Py8szubm55v777zc33XSTffmMGTOMJHPmzJlit+3Ro4dp3rx5mWsBqiNO+QFwcOutt8rd3V3+/v7q0aOHwsLCtHz5ctWuXduh3T333FPqvgrvsHvkkUeKbfPZZ58pLy9PgwYNUl5env3h5eWltm3bas2aNZaOR7p4EbqkIk99ldXw4cP1zTffXPZo3ry5vc3x48f197//XXXr1pWbm5vc3d0VHR0t6eJdiNLF683Wrl2rPn36OIz6Feeuu+5yeN60aVNJv52CrUyLFy/Wn/70J/n5+dmP580337QfiyS1bNlSktSnTx8tWrSoyDtCW7VqpW+//VYPP/ywPvvsM2VkZFR67cCVxik/AA7efvttNWrUSG5ubqpdu7bCw8Mva+Pj46OAgIBS93XixAm5uroqLCys2DbHjh2T9NsH8++5uJT8d19UVJQOHDhQYpvC04t169YtsV1JIiMj7RfrF6WgoECdOnXSkSNHNH78eDVp0kS+vr4qKCjQrbfeqvPnz0uSTp8+rfz8/DLfHVijRg2H54WnAwv3V5SoqChJKrVfSrJ06VL16dNH9957r5544gmFhYXJzc1Nc+fO1bx58+zt2rRpo+TkZM2ePdt+DdgNN9ygcePG6b777pMkjR07Vr6+vnr33Xf16quvytXVVW3atNHzzz9fYp8C1QmBCoCDRo0alfohV9aRntDQUOXn5+vo0aNFBjNJqlmzpiTpgw8+sI/mlEfHjh31z3/+Uxs3bizyOqqsrCytWrVKN954Y4nBzqrvv/9e3377rRYsWKDBgwfblxdeg1YoJCRErq6u+vnnnyutlvDwcDVp0kQrV66034FZXu+++67q1aun999/3+Hn/fuL5iUpMTFRiYmJys7O1saNGzV16lT1799fMTExSkhIkJubm0aOHKmRI0fqzJkzWr16tZ588kl17txZhw8frvB1XsDVhFN+ACpN165dJUlz584ttk3nzp3l5uamffv2KT4+vshHSf7f//t/8vb21rBhwxwmqiw0atQonT59Wk899ZS1gylFYej4/QXur732msNzb29vtW3bVosXLy7y7j9nGT9+vE6fPq3HHnvMfsrzUmfPntXKlSuL3d5ms8nDw8MhTB09erTIu/wKeXp6qm3btnr++eclqci7NIOCgtS7d2898sgjOnXqlMPNCUB1xggVgEpz++23a+DAgXrmmWd07Ngx9ejRQ56entq2bZt8fHw0bNgwxcTE6Omnn9a4ceO0f/9+denSRcHBwTp27Jg2bdokX19fTZ48udjXaNCggd555x0NGDBALVu21MiRI+0Te86bN0/Lly/XqFGj1LdvX0vHkpqaqo0bN162PDQ0VA0aNND111+vBg0a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EagAAAAsIlABAABYRKACAACwiEAFAABgEYEKAADAIgIVAACARQQqAAAAiwhUAAAAFhGoAAAALCJQAQAAWESgAgAAsIhABQAAYBGBCgAAwCICFQAAgEUEKgAAAIsIVAAAABYRqAAAACwiUAEAAFhEoAIAALCIQAUAAGARgQoAAMAiAhUAAIBFBCoAAACLCFQAAAAWEagAAAAsIlABAABYRKACAACwiEAFAABgEYEKAADAIgIVAACARQQqAAAAiwhUAAAAFhGoAAAALCJQAQAAWESgAgAAsIhABQAAYBGBCgAAwCICFQAAgEUEKgAAAIsIVAAAABYRqAAAACwiUAEAAFhEoAIAALCIQAUAAGARgQoAAMAiAhUAAIBFBCoAAACLCFQAAAAWEagAAAAsIlABAABYRKACAACwiEAFAABgEYEKAADAIgIVAACARQQqAAAAiwhUAAAAFhGoAAAALCJQAQAAWESgAgAAsIhABQAAYBGBCgAAwCICFQAAgEUEKgAAAIsIVAAAABYRqAAAACwiUAEAAFhEoAIAALCIQAUAAGARgQoAAMAiAhUAAIBFBCoAAACLCFQAAAAWEagAAAAsIlABAABYRKACAACwiEAFAABgEYEKAADAIgIVAACARQQqAAAAiwhUAAAAFhGoAAAALCJQAQAAWESgAgAAsIhABQAAYBGBCgAAwCICFQAAgEUEKgAAAIsIVAAAABYRqAAAACwiUAEAAFhEoAIAALCIQAUAAGBRhQLVihUr9J///Mf+/J///KeaN2+u/v376/Tp004rDgAAoDqoUKB64oknlJGRIUnasWOHHn/8cXXr1k379+/XyJEjnVogAADA1c6tIhsdOHBAjRs3liQtWbJEPXr00HPPPaetW7eqW7duTi0QAADgalehESoPDw9lZWVJklavXq1OnTpJkkJCQuwjVwAAAH8UFRqhuu222zRy5Ej96U9/0qZNm/T+++9Lkvbs2aPIyEinFggAAHC1q9AI1f/8z//Izc1NH3zwgebOnas6depIkpYvX64uXbo4tUAAAICrnc0YY6q6iGtdRkaGAgMDlZ6eroCAgKouBwAAlEF5Pr8rNEK1detW7dixw/78o48+Us+ePfXkk08qJyenIrsEAACotioUqB588EHt2bNHkrR//37169dPPj4+Wrx4sUaPHu3UAgEAAK52FQpUe/bsUfPm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/Phx+/rq3L8EqmqqVatW5u9//7vDsuuvv96MGTOmiiqqnn4fqAoKCkxYWJiZNm2afdmFCxdMYGCgefXVV40xxpw5c8a4u7ubhQsX2tv88ssvxsXFxaxYseKK1V5dHD9+3Egya9euNcbQx5UhODjYvPHGG/Stk2RmZprY2FizatUq07ZtW3ugon+tmzhxomnWrFmR66p7/3LKrxrKycnRli1b1KlTJ4flnTp10vr166uoqmvDgQMHdPToUYe+9fT0VNu2be19u2XLFuXm5jq0iYiI0I033kj/FyE9PV2SFBISIok+dqb8/HwtXLhQ586dU0JCAn3rJI888oi6d++uDh06OCynf51j7969ioiIUL169dSvXz/t379fUvXvX74cuRr69ddflZ+fr9q1azssr127to4ePVpFVV0bCvuvqL49dOiQvY2Hh4eCg4Mva0P/OzLGaOTIkbrtttt04403SqKPnWHHjh1KSEjQhQsX5Ofnpw8//FCNGze2f6DQtxW3cOFCbd26Vd98881l63jvWnfLLbfo7bff1nXXXadjx47pmWeeUevWrfXDDz9U+/4lUFVjNpvN4bkx5rJlqJiK9C39f7lHH31U3333nf7zn/9cto4+rri4uDht375dZ86c0ZIlSzR48GCtXbvWvp6+rZjDhw9r+PDhWrlypby8vIptR/9WXNeuXe3/36RJEyUkJKhBgwZ66623dOutt0qqvv3LKb9qqGbNmnJ1db0sjR8/fvyyZI/yKbzbpKS+DQsLU05Ojk6fPl1sG0jDhg1TSkqKvvzyS0VGRtqX08fWeXh4qGHDhoqPj9fUqVPVrFkzvfzyy/StRVu2bNHx48d18803y83NTW5ublq7dq1mz54tNzc3e//Qv87j6+urJk2aaO/evdX+/UugqoY8PDx08803a9WqVQ7LV61apdatW1dRVdeGevXqKSwszKFvc3JytHbtWnvf3nzzzXJ3d3dok5aWpu+//57+18W/FB999FEtXbpUX3zxherVq+ewnj52PmOMsrOz6VuL2rdvrx07dmj79u32R3x8vAYMGKDt27erfv369K+TZWdna9euXQoPD6/+79+quBIe1hVOm/Dmm2+anTt3mhEjRhhfX19z8ODBqi7tqpeZmWm2bdtmtm3bZiSZmTNnmm3bttmnnJg2bZoJDAw0S5cuNTt27DD33XdfkbftRkZGmtWrV5utW7eaO++886q4bfdq8NBDD5nAwECzZs0ah1ujs7Ky7G3o44obO3as+eqrr8yBAwfMd999Z5588knj4uJiVq5caYyhb53t0rv8jKF/rXr88cfNmjVrzP79+83GjRtNjx49jL+/v/2zqzr3L4GqGvvnP/9poqOjjYeHh2nRooX9tnSU7MsvvzSSLnsMHjzYGHPx1t2JEyeasLAw4+npadq0aWN27NjhsI/z58+bRx991ISEhBhvb2/To0cPk5qaWgVHc/Upqm8lmfnz59vb0McVN2TIEPu/+9DQUNO+fXt7mDKGvnW23wcq+teawnml3N3dTUREhOnVq5f54Ycf7Ourc//ajDGmasbGAAAArg1cQwUAAGARgQoAAMAiAhUAAIBFBCoAAACLCFQAAAAWEagAAAAsIlABAABYRKACAACwiEAF4A/hjjvu0IgRI6q6DADXKAIVgGolKSlJNptNNptN7u7uql+/vkaNGqVz586VuN3SpUs1ZcqUSq9vzZo1Cg8PV1FfQrFmzRp77Zc+nnrqqUqvC0DlcqvqAgCgvLp06aL58+crNzdX69at0wMPPKBz585p7ty5l7XNzc2Vu7u7QkJCrkhtKSkpuuuuu2Sz2Ypts3v3bgUEBNif+/n5Vei1Co8NQNVjhApAtePp6amwsDDVrVtX/fv314ABA5ScnCxJmjRpkpo3b6558+apfv368vT0lDHmslN+2dnZGj16tOrWrStPT0/FxsbqzTfftK/fuXOnunXrJj8/P9WuXVsDBw7Ur7/+WmpthYGqJLVq1VJYWJj94efnp2+++UYdO3ZUzZo1FRgYqLZt22rr1q0O29lsNr366qtKTEyUr6+vnnnmGUnSxx9/rJtvvlleXl6qX7++Jk+erLy8vDL2JgBnIFABqPa8vb2Vm5trf/7TTz9p0aJFWrJkibZv317kNoMGDdLChQs1e/Zs7dq1S6+++qp9pCgtLU1t27ZV8+bNtXnzZq1YsULHjh1Tnz59Sqzjhx9+0NGjR9W+fftyH0NmZqYGDx6sdevWaePGjYqNjVW3bt2UmZnp0G7ixIlKTEzUjh07NGTIEH322Wf6y1/+oscee0w7d+7Ua6+9pgULFujZZ58tdw0AKo5TfgCqtU2bNum9995zCDE5OTl65513FBoaWuQ2e/bs0aJFi7Rq1Sp16NBBklS/fn37+rlz56pFixZ67rnn7MvmzZununXras+ePbruuuuK3O9HH32kzp07y8vLq8SaIyMjHZ4fOnRId955p8Oy1157TcHBwVq7dq169OhhX96/f38NGTLE/nzgwIEaM2aMBg8ebD+OKVOmaPTo0Zo4cWKJdQBwHgIVgGpn2bJl8vPzU15ennJzc5WYmKhXXnnFvj46OrrYMCVJ27dvl6urq9q2bVvk+i1btujLL78s8tqmffv2lRioHn744VLrX7dunfz9/e3Pg4ODdfz4cU2YMEFffPGFjh07pvz8fGVlZSk1NdVh2/j4+Mtq/eabbxxGpPLz83XhwgVlZWXJx8en1HoAWEegAlDttGvXTnPnzpW7u7siIiIuuzDb19e3xO29vb1LXF9QUKA///nPev755y9bFx4eXuQ2R48e1datW9W9e/dSqpfq1aunoKAgh2VJSUk6ceKEXnrpJUVHR8vT01MJCQnKyclxaPf7YysoKNDkyZPVq1evy16ntJEyAM5DoAJQ7fj6+qphw4YV3r5JkyYqKCjQ2rVr7af8LtWiRQstWbJEMTExcnMr26/JlJQUJSQkqGbNmhWqad26dZozZ466desmSTp8+HCZLoJv0aKFdu/ebak/AFjHRekA/nBiYmI0ePBgDRkyRMnJyTpw4IDWrFmjRYsWSZIeeeQRnTp1Svfdd582bdqk/fv3a+XKlRoyZIjy8/OL3GdKSooSExMrXFPDhg31zjvvaNeuXfr66681YMCAUkfSJGnChAl6++23NWnSJP3www/atWuX3n//fea2Aq4wAhWAP6S5c+eqd+/eevjhh3X99ddr6NCh9slBIyIi9N///lf5+fnq3LmzbrzxRg0fPlyBgYFycbn81+a5c+f0+eeflzpdQknmzZun06dP66abbtLAgQP12GOPqVatWqVu17lzZy1btkyrVq1Sy5Ytdeutt2rmzJmKjo6ucC0Ays9miprOFwBQZkuXLtVTTz2lnTt3VnUpAKoII1QAYJGfn1+RF7AD+ONghAoAAMAiRqgAAAAsIlABAABYRKACAACwiEAFAABgEYEKAADAIgIVAACARQQqAAAAiwhUAAAAFhGoAAAALPr/Rl6ABus9XQ4AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Plot the Prices Paid Of Each Class\n",
    "plt.scatter(titanic['fare'], titanic['class'],  color = 'purple', label='Passenger Paid')\n",
    "plt.ylabel('Class')\n",
    "plt.xlabel('Price / Fare')\n",
    "plt.title('Price Of Each Class')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "abfaf904-c0db-46ed-9904-46cd2c158255",
   "metadata": {},
   "source": [
    "Check which columns contain empty values (NaN, NAN, na). Looks like columns age, embarked, deck, and embarked_town are missing some values.\n",
    "\n",
    "All the other columns are not missing any values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b39a864d-dc8f-4267-9b10-81046bbdcb7f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "survived         0\n",
       "pclass           0\n",
       "sex              0\n",
       "age            177\n",
       "sibsp            0\n",
       "parch            0\n",
       "fare             0\n",
       "embarked         2\n",
       "class            0\n",
       "who              0\n",
       "adult_male       0\n",
       "deck           688\n",
       "embark_town      2\n",
       "alive            0\n",
       "alone            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Count the empty (NaN, NAN, na) values in each column \n",
    "titanic.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "78a97d63-72ad-4420-ad1c-42fd6559333f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0    549\n",
      "1    342\n",
      "Name: survived, dtype: int64\n",
      "\n",
      "3    491\n",
      "1    216\n",
      "2    184\n",
      "Name: pclass, dtype: int64\n",
      "\n",
      "male      577\n",
      "female    314\n",
      "Name: sex, dtype: int64\n",
      "\n",
      "24.00    30\n",
      "22.00    27\n",
      "18.00    26\n",
      "19.00    25\n",
      "28.00    25\n",
      "         ..\n",
      "36.50     1\n",
      "55.50     1\n",
      "0.92      1\n",
      "23.50     1\n",
      "74.00     1\n",
      "Name: age, Length: 88, dtype: int64\n",
      "\n",
      "0    608\n",
      "1    209\n",
      "2     28\n",
      "4     18\n",
      "3     16\n",
      "8      7\n",
      "5      5\n",
      "Name: sibsp, dtype: int64\n",
      "\n",
      "0    678\n",
      "1    118\n",
      "2     80\n",
      "5      5\n",
      "3      5\n",
      "4      4\n",
      "6      1\n",
      "Name: parch, dtype: int64\n",
      "\n",
      "8.0500     43\n",
      "13.0000    42\n",
      "7.8958     38\n",
      "7.7500     34\n",
      "26.0000    31\n",
      "           ..\n",
      "35.0000     1\n",
      "28.5000     1\n",
      "6.2375      1\n",
      "14.0000     1\n",
      "10.5167     1\n",
      "Name: fare, Length: 248, dtype: int64\n",
      "\n",
      "S    644\n",
      "C    168\n",
      "Q     77\n",
      "Name: embarked, dtype: int64\n",
      "\n",
      "Third     491\n",
      "First     216\n",
      "Second    184\n",
      "Name: class, dtype: int64\n",
      "\n",
      "man      537\n",
      "woman    271\n",
      "child     83\n",
      "Name: who, dtype: int64\n",
      "\n",
      "True     537\n",
      "False    354\n",
      "Name: adult_male, dtype: int64\n",
      "\n",
      "C    59\n",
      "B    47\n",
      "D    33\n",
      "E    32\n",
      "A    15\n",
      "F    13\n",
      "G     4\n",
      "Name: deck, dtype: int64\n",
      "\n",
      "Southampton    644\n",
      "Cherbourg      168\n",
      "Queenstown      77\n",
      "Name: embark_town, dtype: int64\n",
      "\n",
      "no     549\n",
      "yes    342\n",
      "Name: alive, dtype: int64\n",
      "\n",
      "True     537\n",
      "False    354\n",
      "Name: alone, dtype: int64\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#Look at all of the values in each column & get a count \n",
    "for val in titanic:\n",
    "   print(titanic[val].value_counts())\n",
    "   print()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c04f0c12-1015-4dd0-98dc-ecf244972ac0",
   "metadata": {},
   "source": [
    "Next, I will drop the redundant columns that are non-numerical and remove rows with missing values.\n",
    "\n",
    "I also decided to drop the column called deck because it’s missing 688 rows of data which means 688/891 = 77.22% of the data is missing for this column."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6371ec9c-0db1-4b8e-b85f-032804e3b3ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop the columns\n",
    "titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'alone', 'adult_male', 'who'], axis=1)\n",
    "\n",
    "#Remove the rows with missing values\n",
    "titanic = titanic.dropna(subset =['embarked', 'age'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e609b2e3-b97e-49e6-821e-4fc4921ec0ee",
   "metadata": {},
   "source": [
    "Now, let’s see the new number of rows and columns in the Titanic data set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "095f9f8f-ffae-46d7-92bb-72d80f808143",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(712, 8)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Count the NEW number of rows and columns in the data set\n",
    "titanic.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "26543200-4ded-452e-a682-b0e55641b782",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "survived      int64\n",
       "pclass        int64\n",
       "sex          object\n",
       "age         float64\n",
       "sibsp         int64\n",
       "parch         int64\n",
       "fare        float64\n",
       "embarked     object\n",
       "dtype: object"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "6be26b57-bffd-41b5-97e3-3f30f6643ea7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['male' 'female']\n",
      "['S' 'C' 'Q']\n"
     ]
    }
   ],
   "source": [
    "#Print the unique values in the columns\n",
    "print(titanic['sex'].unique())\n",
    "print(titanic['embarked'].unique())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "52db6a5e-5218-4250-90ee-96e595a1a540",
   "metadata": {},
   "source": [
    "Change the non-numeric data to numeric data, and print the new values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "524d7fab-eb3c-42da-8ff4-20ab3a451e39",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 0]\n",
      "[2 0 1]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_111/1236119090.py:6: FutureWarning: In a future version, `df.iloc[:, i] = newvals` will attempt to set the values inplace instead of always setting a new array. To retain the old behavior, use either `df[df.columns[i]] = newvals` or, if columns are non-unique, `df.isetitem(i, newvals)`\n",
      "  titanic.iloc[:,2]= labelencoder.fit_transform(titanic.iloc[:,2].values)\n",
      "/tmp/ipykernel_111/1236119090.py:10: FutureWarning: In a future version, `df.iloc[:, i] = newvals` will attempt to set the values inplace instead of always setting a new array. To retain the old behavior, use either `df[df.columns[i]] = newvals` or, if columns are non-unique, `df.isetitem(i, newvals)`\n",
      "  titanic.iloc[:,7]= labelencoder.fit_transform(titanic.iloc[:,7].values)\n"
     ]
    }
   ],
   "source": [
    "#Encoding categorical data values (Transforming object data types to integers)\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "labelencoder = LabelEncoder()\n",
    "\n",
    "#Encode sex column\n",
    "titanic.iloc[:,2]= labelencoder.fit_transform(titanic.iloc[:,2].values)\n",
    "#print(labelencoder.fit_transform(titanic.iloc[:,2].values))\n",
    "\n",
    "#Encode embarked\n",
    "titanic.iloc[:,7]= labelencoder.fit_transform(titanic.iloc[:,7].values)\n",
    "#print(labelencoder.fit_transform(titanic.iloc[:,7].values))\n",
    "\n",
    "#Print the NEW unique values in the columns\n",
    "print(titanic['sex'].unique())\n",
    "print(titanic['embarked'].unique())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fef5846a-2678-449e-aca6-560aca593561",
   "metadata": {},
   "source": [
    "Split the data into independent ‘X’ and dependent ‘Y’ data sets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "3fc77692-bd9b-4da7-8d67-d2a13436ca5c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Split the data into independent 'X' and dependent 'Y' variables\n",
    "X = titanic.iloc[:, 1:8].values \n",
    "Y = titanic.iloc[:, 0].values "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "69c03d87-38f6-45ea-8094-83282efee8fc",
   "metadata": {},
   "source": [
    "Split the data again, this time into 80% training (X_train and Y_train) and 20% testing (X_test and Y_test) data sets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "e88120a4-893f-4837-beda-4ddd6f34f1cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split the dataset into 80% Training set and 20% Testing set\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "2b20d45b-398d-4b9a-9cce-66afd51203dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Feature Scaling\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "sc = StandardScaler()\n",
    "X_train = sc.fit_transform(X_train)\n",
    "X_test = sc.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "619e1ff9-aa24-49a8-bf2b-2a8263e8e3a6",
   "metadata": {},
   "source": [
    "Create a function that has within it many different machine learning models that we can use to make our predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "97565af0-409c-4050-87dd-8a43199435c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a function within many Machine Learning Models\n",
    "def models(X_train,Y_train):\n",
    "  \n",
    "  #Using Logistic Regression Algorithm to the Training Set\n",
    "  from sklearn.linear_model import LogisticRegression\n",
    "  log = LogisticRegression(random_state = 0)\n",
    "  log.fit(X_train, Y_train)\n",
    "  \n",
    "  #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm\n",
    "  from sklearn.neighbors import KNeighborsClassifier\n",
    "  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)\n",
    "  knn.fit(X_train, Y_train)\n",
    "\n",
    "  #Using SVC method of svm class to use Support Vector Machine Algorithm\n",
    "  from sklearn.svm import SVC\n",
    "  svc_lin = SVC(kernel = 'linear', random_state = 0)\n",
    "  svc_lin.fit(X_train, Y_train)\n",
    "\n",
    "  #Using SVC method of svm class to use Kernel SVM Algorithm\n",
    "  from sklearn.svm import SVC\n",
    "  svc_rbf = SVC(kernel = 'rbf', random_state = 0)\n",
    "  svc_rbf.fit(X_train, Y_train)\n",
    "\n",
    "  #Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm\n",
    "  from sklearn.naive_bayes import GaussianNB\n",
    "  gauss = GaussianNB()\n",
    "  gauss.fit(X_train, Y_train)\n",
    "\n",
    "  #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm\n",
    "  from sklearn.tree import DecisionTreeClassifier\n",
    "  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)\n",
    "  tree.fit(X_train, Y_train)\n",
    "\n",
    "  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm\n",
    "  from sklearn.ensemble import RandomForestClassifier\n",
    "  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)\n",
    "  forest.fit(X_train, Y_train)\n",
    "  \n",
    "  #print model accuracy on the training data.\n",
    "  print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))\n",
    "  print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))\n",
    "  print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))\n",
    "  print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))\n",
    "  print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))\n",
    "  print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))\n",
    "  print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))\n",
    "  \n",
    "  return log, knn, svc_lin, svc_rbf, gauss, tree, forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "9a76a595-9ade-482d-84ec-49410fd4ead7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]Logistic Regression Training Accuracy: 0.7978910369068541\n",
      "[1]K Nearest Neighbor Training Accuracy: 0.8664323374340949\n",
      "[2]Support Vector Machine (Linear Classifier) Training Accuracy: 0.7768014059753954\n",
      "[3]Support Vector Machine (RBF Classifier) Training Accuracy: 0.8506151142355008\n",
      "[4]Gaussian Naive Bayes Training Accuracy: 0.8031634446397188\n",
      "[5]Decision Tree Classifier Training Accuracy: 0.9929701230228472\n",
      "[6]Random Forest Classifier Training Accuracy: 0.9753954305799648\n"
     ]
    }
   ],
   "source": [
    "#Get and train all of the models\n",
    "model = models(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "68f418e8-b297-4f2a-ae9c-30a8ac4f6868",
   "metadata": {},
   "source": [
    "Show the confusion matrix and accuracy for all the models on the test data.\n",
    "\n",
    "The model that was most accurate on the test data is the model at position 0, which is the Logistic Regression Model with an accuracy of 81.11%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "48b21399-f7f7-49a9-be5f-8a4f9973240c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[73  9]\n",
      " [18 43]]\n",
      "Model[0] Testing Accuracy = \"0.8111888111888111 !\"\n",
      "\n",
      "[[71 11]\n",
      " [20 41]]\n",
      "Model[1] Testing Accuracy = \"0.7832167832167832 !\"\n",
      "\n",
      "[[70 12]\n",
      " [18 43]]\n",
      "Model[2] Testing Accuracy = \"0.7902097902097902 !\"\n",
      "\n",
      "[[75  7]\n",
      " [22 39]]\n",
      "Model[3] Testing Accuracy = \"0.7972027972027972 !\"\n",
      "\n",
      "[[69 13]\n",
      " [23 38]]\n",
      "Model[4] Testing Accuracy = \"0.7482517482517482 !\"\n",
      "\n",
      "[[60 22]\n",
      " [10 51]]\n",
      "Model[5] Testing Accuracy = \"0.7762237762237763 !\"\n",
      "\n",
      "[[67 15]\n",
      " [13 48]]\n",
      "Model[6] Testing Accuracy = \"0.8041958041958042 !\"\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix \n",
    "for i in range(len(model)):\n",
    "    cm = confusion_matrix(Y_test, model[i].predict(X_test)) \n",
    "    #extracting TN, FP, FN, TP\n",
    "    TN, FP, FN, TP = confusion_matrix(Y_test, model[i].predict(X_test)).ravel()\n",
    "    print(cm)\n",
    "    print('Model[{}] Testing Accuracy = \"{} !\"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))\n",
    "    print()# Print a new line"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a1b8d383-d424-4023-ae01-fde00203063c",
   "metadata": {},
   "source": [
    "The model that I will use to predict if I would’ve survived, will be the model at position 6, the Random Forest Classifier.\n",
    "\n",
    "I chose that model because it did second-best on the training and testing data and has an accuracy of 80.41% on the testing data and 97.53% on the training data.\n",
    "\n",
    "Now we can get the important features.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "19511709-3fc3-41c3-ac5f-eaf416a7375b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>importance</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>feature</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>age</th>\n",
       "      <td>0.300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>fare</th>\n",
       "      <td>0.296</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sex</th>\n",
       "      <td>0.183</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pclass</th>\n",
       "      <td>0.098</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sibsp</th>\n",
       "      <td>0.050</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>parch</th>\n",
       "      <td>0.044</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>embarked</th>\n",
       "      <td>0.030</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          importance\n",
       "feature             \n",
       "age            0.300\n",
       "fare           0.296\n",
       "sex            0.183\n",
       "pclass         0.098\n",
       "sibsp          0.050\n",
       "parch          0.044\n",
       "embarked       0.030"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Get the importance of the features\n",
    "forest = model[6]\n",
    "importances = pd.DataFrame({'feature':titanic.iloc[:, 1:8].columns,'importance':np.round(forest.feature_importances_,3)})\n",
    "importances = importances.sort_values('importance',ascending=False).set_index('feature')\n",
    "importances"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "e28b0a71-fade-488b-80bb-8c68c95e5ddf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: xlabel='feature'>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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eLhefPFFRUZGau7cuc7+N910k6SLgWjGjBnauHGjoqKiJEkNGzbU1q1b9eqrr17VwOLWJaGaNWvK09OzwMjH4cOHC4yQ/F5YWJhatGihP//5zxo1apQmTZrkfK9OnTpu79PHx0f+/v4uGwAAV0vLli2df77086lFixYF2i4tQy9JlSpVUmRkpPN1eHi4AgMDtXv3bknS7t271aFDB5ev06FDB+f7l/x2H5fzww8/qF+/fmrYsKH8/f0VFhYmScrMzCzyWC4tl3+p7ksjLIXZtWuXzp49q65du6pq1arObfHixS6Xnq4Gt0ZYvL29FRERodTUVPXq1cvZnpqaqp49exZ7P5ZlKTc31/k6KipKqampGjVqlLNtw4YNat++vTvlAQBw1fz2jqFLVwAKa8vPz3f5XGFXC37bVpwrDFWqVClWjffee69CQkI0f/581atXT/n5+WrevLnOnTv3h8dyqe5Ll5cKc6nP+++/r+uuu87lPR8fn2LVWFJuXxKKj49XbGysIiMjFRUVpeTkZGVmZmro0KGSLl6qOXTokBYvXixJeuWVV3T99dcrPDxc0sV1WaZNm6bhw4c79zly5EjdfvvteuGFF9SzZ0+tXr1aH374obZu3VoaxwgAgC0uXLigzz//XG3atJEkffvttzp27JjzZ2LTpk21detWDRgwwPmZ7du3q2nTppfdr7e3tyQ555VI0tGjR7V79269+uqr6tixoySV6Odoy5Yt9a9//avQeaKX1kvLzMy8qpd/CuN2YImJidHRo0c1ZcoUZWVlqXnz5lq7dq1CQ0MlSVlZWS5DT/n5+Ro3bpz27dunSpUq6YYbbtDzzz/vvA4oSe3bt9eyZcs0YcIEPf3007rhhhu0fPly1mAxDBOSAcA9Xl5eGj58uGbPni0vLy/9z//8j9q1a+cMME8++aT69u2rW265RV26dNE///lPrVy5Uh9++OFl9xsaGiqHw6E1a9bo7rvvlp+fn6pXr66goCAlJyerbt26yszM1NixY92uedy4cWrRooWGDRumoUOHytvbWx999JEeeOAB1axZU6NHj9aoUaOUn5+v2267TTk5Odq+fbuqVq2qRx55pETnqThKNOl22LBhzglCv5eSkuLyevjw4S6jKUXp06eP+vTpU5JyAABlUEX4paNy5coaM2aM+vXrp//+97+67bbb9Nprrznf/9Of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xWct/frrr/rkk0/UuXNnrVq1Sj/99JOmTp2q6dOn212eLTw9PdWtWzft3r2bwAIUZdGiRXaXgDKECZOFCwgIcAa5gIAAm6sx28aNG7V69Wrdeuut8vDwUGhoqLp27Sp/f38lJCQ4b5WvaC6tT/P7hx+WJVwSAoAy5MyZM8rPz3feirp//369++67atq0Kc+f0sVLhV999ZUaNGigBg0a6M0331SHDh20b98+3XTTTRX2lvgNGzZozJgxevbZZxUREVHgVuaycImVERYARlq0aJGqVq2qBx54wKX97bff1unTp/XII4/YVJm9evbs6fL8qXbt2vH8qd9o0qSJvv32WzVo0EA333yzXn31VTVo0EDz5s1T3bp17S7PNpcu0993330ul13L0iVWAgsAIz3//POFLjVfu3ZtDRkypMIGli+++MK5Wuk777yj4OBgpaena8WKFZo4cWKFDyxxcXHKysqSdPHOl27duukf//iHvL299frrr9tcnX3Kw+VWLgkBMJKvr6+++eYbNWjQwKV9//79atq0qc6cOWNPYTarXLmyvvnmG11//fXq27evbrrpJj3zzDM6ePCgmjRpUmEveRTGsiydOXPGeb5q1qxpd0m4AizND8BItWvXdj6H6re+/PJLBQUF2VCRGXj+1B9buHChmjdvLl9fX1WvXl0DBgzQu+++a3dZRjh9+rS++eYbffXVVy5bWcAlIQBGevDBBzVixAhVq1ZNt99+uyTp448/1siRI/Xggw/aXJ19Jk6cqH79+mnUqFHq0qULz5/6naefflozZ87U8OHDnecmLS1No0aN0v79+zV16lSbK7RHeVifhktCAIx07tw5xcbG6u2331alShd/t8rLy9MjjzyiefPmydvb2+YK7ZOdne18/pSHx8WB8k8//VT+/v4KDw+3uTp71axZU3PmzNFDDz3k0r506VINHz5cR44csakyez388MPav3+/EhMTC12fpizc7k1gAWC0PXv2KD09XX5+fmrZsqVCQ0PtLgkGq169uj799FM1atTIpf27775TmzZtdOzYMXsKs1ndunW1evVqtWnTRv7+/vr888/VuHFjvffee3rxxRe1detWu0v8Q8xhAWCshQsXqlevXoqNjVWfPn3Uo0cPLViwwO6yYLD+/fsrKSmpQHtycrIefvhhGyoyw6lTp1S7dm1JUo0aNfTzzz9L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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Visualize the importance\n",
    "importances.plot.bar()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "656901b9-6754-441f-962a-8bdf821170b0",
   "metadata": {},
   "source": [
    "Print the Random Forest Classifier Model predictions for each passenger and, below it, print the actual values. Remember ‘1’ means the passenger survived and ‘0’ means the passenger did not survive.\n",
    "\n",
    "By printing both we can visually see how well the model did on the test data, but remember the model was 80.41% accurate on the testing data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "bacee5c1-11a3-4d05-99d3-6eda2f84dc38",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 1 0 0 0 1 0 0 1 1 1 1 0 0 1 0 1 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1\n",
      " 1 1 0 0 0 1 0 1 0 1 0 0 1 1 0 1 0 1 0 0 1 1 1 0 0 0 1 0 0 1 0 1 1 1 1 1 1\n",
      " 0 0 1 0 0 0 0 1 0 1 1 0 0 0 1 0 0 0 1 1 1 0 1 1 0 0 0 1 0 0 0 0 1 0 0 0 1\n",
      " 0 1 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 1 1 1 1 1 0 0 0 0 0 1]\n",
      "\n",
      "[0 0 1 0 0 0 1 0 0 0 1 1 1 0 0 1 0 1 1 0 0 1 1 1 0 0 0 0 1 0 0 0 0 1 1 0 1\n",
      " 1 1 1 1 1 0 0 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 1 1 0 1 0 1 1 1\n",
      " 0 0 1 1 0 0 0 1 1 1 1 0 0 0 1 0 0 0 1 1 1 0 1 1 0 1 0 1 0 0 0 0 1 0 0 0 0\n",
      " 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 1]\n"
     ]
    }
   ],
   "source": [
    "#Print Prediction of Random Forest Classifier model\n",
    "pred = model[6].predict(X_test)\n",
    "print(pred)\n",
    "\n",
    "#Print a space\n",
    "print()\n",
    "\n",
    "#Print the actual values\n",
    "print(Y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6ff456de-c9d0-45be-a3d4-e55aa5b1b14f",
   "metadata": {},
   "source": [
    "Now that we have analyzed the data, created our models, and chosen a model to predict who would’ve survived the Titanic, let’s test and see if I would have survived.\n",
    "\n",
    "Putting those values in an array gives me [3,1,21,0, 0, 0, 1]. But, to put this into the prediction method of the model, it must be a list of lists or 2D array, for example [[3,1,21,0, 0, 0, 1]]."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "2aabd53f-2f91-4c6c-858e-436984434a2d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]\n",
      "Oh no! You didn't make it\n"
     ]
    }
   ],
   "source": [
    "my_survival = [[3,1,21,0, 0, 0, 1]]\n",
    "#Print Prediction of Random Forest Classifier model\n",
    "pred = model[6].predict(my_survival)\n",
    "print(pred)\n",
    "\n",
    "if pred == 0:\n",
    "    print(\"Oh no! You didn't make it\")\n",
    "else:\n",
    "    print('Nice! You survived')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "abc1f328-30ae-4ccd-a4c7-22172f2214b2",
   "metadata": {},
   "source": [
    "Conclusion:\n",
    "That is it, we are done creating your program to predict if a passenger would survive the Titanic or not!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53135991-5e9f-4f64-9251-f0b8f05a4f6c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
