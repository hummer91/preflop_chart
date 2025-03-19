// chart.js에 있던 데이터를 불러와야 하므로,
// chart.js 내용을 script.js 맨 위에 그대로 복사해 넣어도 좋아.

let currentHand = "";

function getRandomHand() {
	const hands = Object.keys(preflopChart);
	return hands[Math.floor(Math.random() * hands.length)];
}

function loadNewHand() {
	currentHand = getRandomHand();
	document.getElementById("hand").innerText = `핸드: ${currentHand}`;
	document.getElementById("feedback").innerText = "";
}

function selectAction(action) {
	const correctAction = preflopChart[currentHand];
	const feedback = document.getElementById("feedback");

	if (action === correctAction) {
		feedback.innerText = "정답! 🎉";
	} else {
		feedback.innerText = `오답! 정답은 "${correctAction}"입니다.`;
	}

	setTimeout(loadNewHand, 2000); // 2초 뒤 다음 문제
}

// 페이지 로드시 첫 문제 표시
window.onload = loadNewHand;
