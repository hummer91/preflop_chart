import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 준비 (예시)
# 실제로는 많은 양의 핸드 히스토리 데이터 필요
def generate_poker_dataset(num_samples=10000):
    """포커 데이터셋 생성 (실제로는 실제 게임 데이터 사용)"""
    # 카드 랭크 및 수트
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['c', 'd', 'h', 's']
    
    # 데이터 프레임 초기화
    data = []
    
    for _ in range(num_samples):
        # 랜덤 카드 2장 생성
        card1_rank = np.random.choice(ranks)
        card1_suit = np.random.choice(suits)
        card2_rank = np.random.choice(ranks)
        card2_suit = np.random.choice(suits)
        
        # 중복 카드 방지
        while (card1_rank == card2_rank and card1_suit == card2_suit):
            card2_rank = np.random.choice(ranks)
            card2_suit = np.random.choice(suits)
        
        # 카드 순서 정렬 (높은 카드가 먼저 오도록)
        rank_values = {r: i for i, r in enumerate(ranks)}
        if rank_values[card1_rank] < rank_values[card2_rank]:
            card1_rank, card2_rank = card2_rank, card1_rank
            card1_suit, card2_suit = card2_suit, card1_suit
        
        # 포지션 랜덤 생성 (6인 테이블: BTN, SB, BB, UTG, MP, CO)
        positions = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']
        position = np.random.choice(positions)
        
        # 액션 전 플레이어 수
        players_before = np.random.randint(0, 6)
        
        # 액션 (0: 폴드, 1: 콜, 2: 레이즈)
        # 실제로는 이 부분이 더 복잡함 - 실제 게임 데이터 필요
        # 여기서는 간단한 규칙으로 생성 (실제 사용 시 실제 게임 데이터로 대체)
        action = 0  # 기본값: 폴드
        
        # 강한 핸드는 레이즈할 확률 높음
        if (card1_rank == card2_rank) or (card1_rank in 'AKQ' and card2_rank in 'AKQ'):
            action = np.random.choice([0, 1, 2], p=[0.1, 0.3, 0.6])
        # 중간 강도 핸드는 콜/레이즈 확률 중간
        elif (card1_rank in 'AKQJT' or card2_rank in 'AKQJT'):
            action = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        # 약한 핸드는 폴드 확률 높음
        else:
            action = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        
        # 포지션에 따른 조정
        if position in ['BTN', 'CO']:  # 레이트 포지션은 더 공격적
            if action == 0 and np.random.random() < 0.3:  # 30% 확률로 폴드를 콜/레이즈로 변경
                action = np.random.choice([1, 2])
        elif position in ['UTG']:  # 얼리 포지션은 더 타이트하게
            if action != 0 and np.random.random() < 0.3:  # 30% 확률로 콜/레이즈를 폴드로 변경
                action = 0
        
        # 데이터 추가
        data.append({
            'card1_rank': card1_rank,
            'card1_suit': card1_suit,
            'card2_rank': card2_rank,
            'card2_suit': card2_suit,
            'position': position,
            'players_before': players_before,
            'suited': 1 if card1_suit == card2_suit else 0,
            'pair': 1 if card1_rank == card2_rank else 0,
            'action': action
        })
    
    return pd.DataFrame(data)

# 데이터셋 생성
df = generate_poker_dataset(50000)

# 2. 특성 엔지니어링
def create_features(df):
    # 랭크를 숫자로 변환
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                  '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    df['card1_value'] = df['card1_rank'].map(rank_values)
    df['card2_value'] = df['card2_rank'].map(rank_values)
    
    # 카드 차이 (연결된 카드 여부 확인)
    df['card_gap'] = df['card1_value'] - df['card2_value']
    
    # 핸드 타입 특성
    df['connected'] = (df['card_gap'] == 1).astype(int)  # 연결된 카드
    df['one_gap'] = (df['card_gap'] == 2).astype(int)   # 1갭 카드
    
    # 포지션을 숫자로 변환
    position_values = {'UTG': 0, 'MP': 1, 'CO': 2, 'BTN': 3, 'SB': 4, 'BB': 5}
    df['position_value'] = df['position'].map(position_values)
    
    # 원핫 인코딩
    df = pd.get_dummies(df, columns=['position'])
    
    return df

df = create_features(df)

# 3. 모델 학습
# 특성과 타깃 분리
X = df.drop(['action', 'card1_rank', 'card1_suit', 'card2_rank', 'card2_suit', 'position_value'], axis=1)
y = df['action']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 모델 평가
y_pred = model.predict(X_test)
print(f"정확도: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 5. 프리플랍 차트 생성
def generate_preflop_chart(model):
    """머신러닝 모델을 사용하여 프리플랍 차트 생성"""
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    
    # 포지션별 차트 생성
    positions = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']
    charts = {}
    
    for position in positions:
        # 빈 차트 초기화 (0: 폴드, 1: 콜, 2: 레이즈)
        chart = np.zeros((len(ranks), len(ranks)))
        
        # 각 핸드 조합에 대해 예측
        for i, rank1 in enumerate(ranks):
            for j, rank2 in enumerate(ranks):
                if i < j:  # 수트 핸드 (예: AKs)
                    # 특성 생성
                    features = {
                        'card1_value': rank_values[rank1],
                        'card2_value': rank_values[rank2],
                        'suited': 1,
                        'pair': 0,
                        'players_before': 2,  # 평균값으로 가정
                        'card_gap': rank_values[rank1] - rank_values[rank2],
                        'connected': 1 if rank_values[rank1] - rank_values[rank2] == 1 else 0,
                        'one_gap': 1 if rank_values[rank1] - rank_values[rank2] == 2 else 0,
                    }
                    
                    # 포지션 원핫 인코딩 추가
                    for pos in positions:
                        features[f'position_{pos}'] = 1 if pos == position else 0
                    
                    # 모델 예측
                    features_df = pd.DataFrame([features])
                    prediction = model.predict(features_df)[0]
                    chart[i, j] = prediction
                    
                elif i > j:  # 오프수트 핸드 (예: AKo)
                    # 특성 생성 (위와 유사하나 suited=0)
                    features = {
                        'card1_value': rank_values[rank2],
                        'card2_value': rank_values[rank1],
                        'suited': 0,
                        'pair': 0,
                        'players_before': 2,
                        'card_gap': rank_values[rank2] - rank_values[rank1],
                        'connected': 1 if rank_values[rank2] - rank_values[rank1] == 1 else 0,
                        'one_gap': 1 if rank_values[rank2] - rank_values[rank1] == 2 else 0,
                    }
                    
                    # 포지션 원핫 인코딩 추가
                    for pos in positions:
                        features[f'position_{pos}'] = 1 if pos == position else 0
                    
                    # 모델 예측
                    features_df = pd.DataFrame([features])
                    prediction = model.predict(features_df)[0]
                    chart[i, j] = prediction
                    
                else:  # 포켓 페어 (예: AA)
                    # 특성 생성
                    features = {
                        'card1_value': rank_values[rank1],
                        'card2_value': rank_values[rank1],
                        'suited': 0,  # 포켓 페어는 suited가 아님
                        'pair': 1,
                        'players_before': 2,
                        'card_gap': 0,
                        'connected': 0,
                        'one_gap': 0,
                    }
                    
                    # 포지션 원핫 인코딩 추가
                    for pos in positions:
                        features[f'position_{pos}'] = 1 if pos == position else 0
                    
                    # 모델 예측
                    features_df = pd.DataFrame([features])
                    prediction = model.predict(features_df)[0]
                    chart[i, j] = prediction
        
        charts[position] = chart
    
    return charts, ranks

# 차트 생성
charts, ranks = generate_preflop_chart(model)

# 차트 시각화
def plot_preflop_chart(chart, ranks, position):
    """프리플랍 차트 시각화"""
    plt.figure(figsize=(10, 8))
    
    # 색상 매핑 (0: 폴드-빨강, 1: 콜-노랑, 2: 레이즈-초록)
    cmap = sns.color_palette(["red", "yellow", "green"], as_cmap=True)
    
    # 히트맵 그리기
    ax = sns.heatmap(chart, cmap=cmap, linewidths=0.5, linecolor='gray',
                     xticklabels=ranks, yticklabels=ranks)
    
    # 차트 꾸미기
    plt.title(f'6맥스 홀덤 프리플랍 차트 - {position} 포지션')
    plt.xlabel('두 번째 카드')
    plt.ylabel('첫 번째 카드')
    
    # 값에 따라 텍스트 표시
    for i in range(len(ranks)):
        for j in range(len(ranks)):
            if chart[i, j] == 0:
                text = 'F'  # 폴드
            elif chart[i, j] == 1:
                text = 'C'  # 콜
            else:
                text = 'R'  # 레이즈
            
            # 포켓 페어는 'AA', 'KK' 등으로 표시
            if i == j:
                hand_text = f'{ranks[i]}{ranks[i]}'
            # 수트 핸드는 'AKs'와 같이 표시
            elif i < j:
                hand_text = f'{ranks[i]}{ranks[j]}s'
            # 오프수트 핸드는 'AKo'와 같이 표시
            else:
                hand_text = f'{ranks[j]}{ranks[i]}o'
            
            ax.text(j + 0.5, i + 0.5, f'{hand_text}\n{text}', 
                    ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    return plt

# 각 포지션별 차트 시각화
for position, chart in charts.items():
    plt = plot_preflop_chart(chart, ranks, position)
    plt.savefig(f'preflop_chart_{position}.png')
    plt.close()

print("모든 포지션별 프리플랍 차트가 생성되었습니다.")