import numpy as np
import pandas as pd
import itertools
import random
from collections import Counter

# 카드 덱 생성
def create_deck():
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['h', 'd', 'c', 's']
    return [r+s for r in ranks for s in suits]

# 핸드 승률 계산 (몬테카를로 시뮬레이션)
def calculate_hand_equity(hand, num_opponents=5, trials=1000):
    """
    특정 시작 핸드에 대한 승률 계산 (몬테카를로 시뮬레이션)
    """
    wins = 0
    
    for _ in range(trials):
        # 덱 생성 및 셔플
        deck = create_deck()
        for card in hand:
            deck.remove(card)
        random.shuffle(deck)
        
        # 상대방 핸드 배분
        opponent_hands = []
        for i in range(num_opponents):
            opponent_hands.append([deck.pop(), deck.pop()])
        
        # 커뮤니티 카드 5장
        community = [deck.pop() for _ in range(5)]
        
        # 내 핸드 강도
        my_strength = evaluate_hand_strength(hand + community)
        
        # 상대방 핸드 강도
        opponent_strengths = [evaluate_hand_strength(opp_hand + community) for opp_hand in opponent_hands]
        
        # 승리 여부 확인
        if my_strength > max(opponent_strengths):
            wins += 1
        elif my_strength == max(opponent_strengths):
            wins += 0.5  # 동점 시 0.5점
    
    return wins / trials

# 간단한 핸드 강도 평가 함수 (실제로는 더 정교한 알고리즘 필요)
def evaluate_hand_strength(cards):
    """
    매우 간단화된 핸드 평가 함수 (데모용)
    실제로는 정확한 핸드 랭킹 알고리즘 필요
    """
    # 랭크와 수트 분리
    ranks = [card[0] for card in cards]
    suits = [card[1] for card in cards]
    
    # 랭크 값으로 변환
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    values = [rank_values[r] for r in ranks]
    
    # 랭크 카운트
    rank_counts = Counter(ranks)
    
    # 수트 카운트
    suit_counts = Counter(suits)
    
    # 플러시 체크
    flush = max(suit_counts.values()) >= 5
    
    # 스트레이트 체크 (간단한 구현)
    values_sorted = sorted(set(values), reverse=True)
    straight = False
    for i in range(len(values_sorted) - 4):
        if values_sorted[i] - values_sorted[i+4] == 4:
            straight = True
            break
    
    # 페어, 트립스, 쿼드 체크
    pairs = sum(1 for count in rank_counts.values() if count >= 2)
    trips = sum(1 for count in rank_counts.values() if count >= 3)
    quads = sum(1 for count in rank_counts.values() if count >= 4)
    
    # 핸드 강도 점수 계산 (간단화된 버전)
    # 실제로는 정확한 핸드 랭킹 알고리즘 사용 필요
    if straight and flush:
        return 8  # 스트레이트 플러시
    elif quads:
        return 7  # 포카드
    elif trips and pairs >= 2:
        return 6  # 풀하우스
    elif flush:
        return 5  # 플러시
    elif straight:
        return 4  # 스트레이트
    elif trips:
        return 3  # 트립스
    elif pairs >= 2:
        return 2  # 투페어
    elif pairs == 1:
        return 1  # 원페어
    else:
        return 0  # 하이카드

# 모든 가능한 시작 핸드 생성
def generate_all_starting_hands():
    """모든 가능한 시작 핸드 생성"""
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['h', 'd', 'c', 's']
    
    all_cards = [r+s for r in ranks for s in suits]
    return list(itertools.combinations(all_cards, 2))

# 핸드 유형 분류
def classify_hand(hand):
    """핸드 유형 분류 (페어, 수트, 오프수트)"""
    rank1, suit1 = hand[0][0], hand[0][1]
    rank2, suit2 = hand[1][0], hand[1][1]
    
    if rank1 == rank2:
        return f"{rank1}{rank2}"  # 페어 (예: AA, KK)
    elif suit1 == suit2:
        return f"{rank1}{rank2}s"  # 수트 (예: AKs, JTs)
    else:
        return f"{rank1}{rank2}o"  # 오프수트 (예: AKo, JTo)

# 프리플랍 차트 생성 시뮬레이션
def generate_preflop_chart(num_opponents=5, position="BTN", trials_per_hand=1000):
    """
    특정 포지션 및 상대 수에 대한 프리플랍 차트 생성
    """
    # 모든 가능한 시작 핸드
    all_hands = generate_all_starting_hands()
    
    # 핸드별 승률 및 분류 저장
    hand_data = []
    
    # 시뮬레이션 (실제 구현에서는 모든 핸드에 대해 계산, 여기서는 일부만)
    sample_size = min(200, len(all_hands))  # 시간 절약을 위해 샘플만 사용
    for hand in random.sample(all_hands, sample_size):
        equity = calculate_hand_equity(hand, num_opponents, trials_per_hand)
        hand_type = classify_hand(hand)
        
        hand_data.append({
            'hand': hand,
            'hand_type': hand_type,
            'equity': equity
        })
    
    # 핸드 유형별 평균 승률 계산
    hand_type_equities = {}
    for data in hand_data:
        hand_type = data['hand_type']
        if hand_type not in hand_type_equities:
            hand_type_equities[hand_type] = []
        hand_type_equities[hand_type].append(data['equity'])
    
    # 평균 승률 계산
    hand_type_avg = {ht: sum(equities)/len(equities) 
                     for ht, equities in hand_type_equities.items()}
    
    # 승률에 따른 액션 결정
    # 예: 승률 55% 이상 -> 레이즈, 45-55% -> 콜, 45% 미만 -> 폴드
    # 포지션에 따라 임계값 조정
    thresholds = {
        "UTG": (0.50, 0.58),  # 폴드 < 50% < 콜 < 58% < 레이즈
        "MP": (0.48, 0.56),
        "CO": (0.45, 0.54),
        "BTN": (0.42, 0.52),
        "SB": (0.44, 0.53),
        "BB": (0.40, 0.50)
    }
    
    fold_threshold, raise_threshold = thresholds.get(position, (0.45, 0.55))
    
    # 차트 결과 생성
    chart = {}
    for hand_type, avg_equity in hand_type_avg.items():
        if avg_equity >= raise_threshold:
            action = "R"  # 레이즈
        elif avg_equity >= fold_threshold:
            action = "C"  # 콜
        else:
            action = "F"  # 폴드
        chart[hand_type] = action
    
    return chart

# 메인 시뮬레이션 함수
def main():
    # 6맥스 테이블 포지션별 차트 생성
    positions = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
    charts = {}
    
    for position in positions:
        print(f"{position} 포지션 차트 생성 중...")
        # 실제 구현에서는 trials_per_hand를 더 높게 설정 (예: 10000)
        # 여기서는 데모를 위해 낮게 설정
        chart = generate_preflop_chart(
            num_opponents=5,  # 6맥스 테이블 (자신 제외 5명)
            position=position,
            trials_per_hand=100  # 데모를 위해 낮게 설정
        )
        charts[position] = chart
        
        # 결과 출력
        print(f"{position} 포지션 차트 (일부 핸드 표시):")
        for hand_type, action in sorted(list(chart.items()))[:10]:
            print(f"{hand_type}: {action}")
        print()
    
    # 결과를 CSV 파일로 저장
    for position, chart in charts.items():
        df = pd.DataFrame(list(chart.items()), columns=['Hand', 'Action'])
        df.to_csv(f'preflop_chart_{position}.csv', index=False)
    
    print("모든 포지션별 프리플랍 차트가 생성되었습니다.")

if __name__ == "__main__":
    main()