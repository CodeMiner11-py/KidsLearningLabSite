// learn.js — Kids Learning Lab "Learn" page (Duolingo-style AI courses)
import { db, auth, rtdb } from "./firebase.js";
import { notifyUser } from "./notifications.js";
import { checkStreakBadges, checkLessonBadges, checkGameBadge, checkCourseBadges, checkUnitCompletionBadges, checkCourseCompletionBadges, pauseCelebrations, resumeCelebrations } from "./badges.js";
import { limits, isPremium } from "./premium.js";
import { openPaywall } from "./paywall.js";
import { normalizeShopFields, hasBankedPremiumPerks, onShopStateChange, shopTodayStr } from "./shop.js";
import { openXpShop } from "./shopUI.js";
import { recordGameHistory, getGameHistoryContext } from "./gameHistory.js";
import { hostSession, joinSession, listenToSession, updateSession, cancelSession, claimField, renderJoinQr, scanJoinCode } from "./multiplayer.js";
import { identifyScannedCode, GAME_SESSION_PATHS } from "./qrRouting.js";
import { sendFriendRequestToUid } from "./friends.js";
import { maybeShowOverlay } from "./firstTimeOverlays.js";
import {
  createRoom, joinRoom, listenToRoom, startRound, reportCorrectAnswer,
  endRound, resetRoomForNewRound, leaveRoom, cancelRoom,
} from "./groupGame.js";
import {
  doc, getDoc, setDoc, updateDoc, deleteDoc, collection, getDocs, query, orderBy, where, onSnapshot, serverTimestamp, increment
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";
import {
  ref as rtdbRef, get as rtdbGet, set as rtdbSet
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

const LEARN_WORKER_URL = 'https://kidslearninglabtextworker.nameless-cherry-998c.workers.dev/';
const EMAIL_WORKER_URL = 'https://emailworkerkidslearninglabanyhtmlnonspecific.nameless-cherry-998c.workers.dev/';
const DIAGRAM_FEEDBACK_TO_EMAIL = 'ebuddhisagar@gmail.com';
const UNITS_PER_COURSE = 10;
const LESSONS_PER_UNIT = 15; // index 14 (the 15th) is always the unit review
const MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
  'August', 'September', 'October', 'November', 'December'];

// ---- Course colors: picked randomly client-side, no AI involved ----
// A curated set of vivid, readable colors spanning the wheel (blue included).
const COURSE_COLORS = [
  '#1E6FE0', '#2F80ED', '#3B82F6', '#1D4ED8', '#2563EB', '#0EA5E9', '#0284C7', '#0891B2',
  '#06B6D4', '#0D9488', '#14B8A6', '#059669', '#10B981', '#22C55E', '#16A34A', '#65A30D',
  '#84CC16', '#4D7C0F', '#15803D', '#047857', '#0F766E', '#155E75', '#075985', '#1E40AF',
  '#4338CA', '#4F46E5', '#6366F1', '#7C3AED', '#8B5CF6', '#9333EA', '#A855F7', '#A21CAF',
  '#C026D3', '#D946EF', '#DB2777', '#EC4899', '#F472B6', '#E11D48', '#F43F5E', '#FB7185',
  '#DC2626', '#EF4444', '#F87171', '#EA580C', '#F97316', '#FB923C', '#D97706', '#F59E0B',
  '#FBBF24', '#CA8A04', '#EAB308', '#A16207', '#3F6212', '#166534', '#065F46', '#134E4A',
  '#164E63', '#1E3A8A', '#312E81', '#581C87', '#701A75', '#831843', '#9F1239', '#7F1D1D',
  '#7C2D12', '#78350F', '#713F12', '#365314', '#14532D', '#064E3B', '#0C4A6E', '#1E1B4B',
  '#4C1D95', '#5B21B6', '#6D28D9', '#7E22CE', '#86198F', '#A3E635', '#FACC15', '#FDE047',
  '#38BDF8', '#22D3EE', '#2DD4BF', '#34D399', '#4ADE80', '#FCA5A5', '#FDBA74', '#FCD34D',
  '#F97066', '#EF6820', '#F79009', '#2E90FA', '#53B1FD', '#7A5AF8', '#9E77ED', '#EE46BC',
  '#F63D68', '#12B76A', '#17B26A', '#F04438', '#B42318', '#D0417E', '#6941C6', '#175CD3',
];

// Picks a random course color from the curated palette — simple and no AI involved.
function pickRandomCourseColor() {
  return COURSE_COLORS[Math.floor(Math.random() * COURSE_COLORS.length)];
}

// ---- "I don't know what to learn" quiz — 30-question pool ----
// General-audience, free-text prompts (not aimed specifically at kids, but
// fine for anyone). 5 are picked at random each time the quiz opens, and
// the free-text answers get sent to the worker so the AI can propose a
// course topic. Kept intentionally light/open-ended rather than "what do
// you want to learn" — the point is to surface interests indirectly.
const IDK_QUIZ_QUESTIONS = [
  "What's your favorite animal?",
  "What's your favorite color?",
  "Do you like reading?",
  "Would you rather explore outer space or the deep ocean?",
  "What's your favorite season, and why?",
  "Do you prefer mountains or the beach?",
  "What's a hobby you'd like to try someday?",
  "Do you like solving puzzles or riddles?",
  "What's your favorite kind of music?",
  "Would you rather build something new or take something apart to see how it works?",
  "Do you enjoy cooking or baking?",
  "What's your dream vacation spot?",
  "Are you more curious about history or about science?",
  "Would you rather read a mystery story or a fantasy story?",
  "What's something you're curious about but have never looked into?",
  "Do you enjoy playing sports, or watching them more?",
  "Would you rather learn a new language or a new instrument?",
  "What's your favorite way to spend a rainy day?",
  "Do you like animals more, or plants more?",
  "Would you rather visit a museum or a theme park?",
  "What's a skill you wish you were better at?",
  "Do you enjoy drawing, painting, or other art?",
  "Would you rather explore ancient ruins or a futuristic city?",
  "What's a subject you loved learning about in school?",
  "Do you like board games or video games more?",
  "Would you rather grow a garden or build a robot?",
  "What's a movie or show that made you curious about something?",
  "Do you prefer working with your hands, or thinking things through on paper?",
  "What's a place in the world you'd love to know more about?",
  "Would you rather learn about space, the ocean, or the human body?",
];

// Fisher-Yates shuffle, used to pick 5 non-repeating questions per attempt.
function shuffledCopy(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const IDK_QUIZ_LENGTH = 5;
let idkQuizQuestions = [];  // this attempt's 5 questions
let idkQuizIndex = 0;
let idkQuizAnswers = [];    // [{ question, answer }, ...]

// ---- Correct-answer sound ----
function playCorrectSound() {
  new Audio('./correct.mp3').play().catch(() => {});
}

// ---- Wrong-answer sound (3x volume boost via Web Audio gain — <audio>.volume caps at 1) ----
const wrongSoundCtx = new (window.AudioContext || window.webkitAudioContext)();
async function playWrongSound() {
  if (wrongSoundCtx.state === 'suspended') await wrongSoundCtx.resume();
  const audio = new Audio('./uide.mp3');
  const source = wrongSoundCtx.createMediaElementSource(audio);
  const gainNode = wrongSoundCtx.createGain();
  gainNode.gain.value = 3; // 3x the normal (already-maxed) volume
  source.connect(gainNode).connect(wrongSoundCtx.destination);
  audio.play().catch(() => {});
}

// ---- DOM refs ----
const learnStreakBtn = document.getElementById('learnStreakBtn');
const learnStreakCount = document.getElementById('learnStreakCount');
const learnCoursesBtn = document.getElementById('learnCoursesBtn');

const learnEmptyState = document.getElementById('learnEmptyState');
const learnSkeleton = document.getElementById('learnSkeleton');
const learnCourseHome = document.getElementById('learnCourseHome');
const learnCourseTitle = document.getElementById('learnCourseTitle');
const learnCourseStatus = document.getElementById('learnCourseStatus');
const reviewWrongAnswersBtn = document.getElementById('reviewWrongAnswersBtn');
const reviewWrongAnswersLabel = document.getElementById('reviewWrongAnswersLabel');
const learnCreateFirstBtn = document.getElementById('learnCreateFirstBtn');

// ---- Review Page DOM refs ----
const reviewPageFullWeakBtn = document.getElementById('reviewPageFullWeakBtn');
const reviewAiSummary = document.getElementById('reviewAiSummary');
const reviewAiSummaryText = document.getElementById('reviewAiSummaryText');
const reviewPageFullAllBtn = document.getElementById('reviewPageFullAllBtn');
const reviewWeakSpotsList = document.getElementById('reviewWeakSpotsList');
const reviewStrengthsList = document.getElementById('reviewStrengthsList');
const reviewPageDailyLessonBtn = document.getElementById('reviewPageDailyLessonBtn');
const reviewPageDailyLessonTitle = document.getElementById('reviewPageDailyLessonTitle');
const reviewPageUsesNote = document.getElementById('reviewPageUsesNote');
// Quick tap-press feedback on the Review page's primary buttons.
[reviewPageFullWeakBtn, reviewPageFullAllBtn, reviewPageDailyLessonBtn].forEach((btn) => {
  btn?.addEventListener('pointerdown', () => window.KLLAnim?.tapBounce(btn));
});

const learnUnitsList = document.getElementById('learnUnitsList');
const learnCourseDescription = document.getElementById('learnCourseDescription');
const learnUnitPathView = document.getElementById('learnUnitPathView');
const unitPathBackBtn = document.getElementById('unitPathBackBtn');
const unitPathTitle = document.getElementById('unitPathTitle');
const unitPathDescription = document.getElementById('unitPathDescription');
const learnLessonPath = document.getElementById('learnLessonPath');

const learnCourseReviewCard = document.getElementById('learnCourseReviewCard');
const learnCourseReviewTitle = document.getElementById('learnCourseReviewTitle');
const learnCourseReviewSub = document.getElementById('learnCourseReviewSub');
const learnCourseReviewBtn = document.getElementById('learnCourseReviewBtn');
const courseReviewErrorModalOverlay = document.getElementById('courseReviewErrorModalOverlay');
const courseReviewErrorCloseBtn = document.getElementById('courseReviewErrorCloseBtn');

const streakModalOverlay = document.getElementById('streakModalOverlay');
const streakModalCloseBtn = document.getElementById('streakModalCloseBtn');
const streakModalShopBtn = document.getElementById('streakModalShopBtn');
const streakModalCount = document.getElementById('streakModalCount');
const streakModalMissed = document.getElementById('streakModalMissed');
const streakPassCountLabel = document.getElementById('streakPassCountLabel');
const streakPassPlural = document.getElementById('streakPassPlural');
// Post-lesson streak celebration — its own full-screen overlay, separate
// from the streak-calendar modal above (which is only for the manual
// "view my streak" button).
const lessonStreakOverlay = document.getElementById('lessonStreakOverlay');
const lessonStreakContainer = document.getElementById('lessonStreakContainer');
const lessonStreakCountEl = document.getElementById('lessonStreakCount');
const lessonStreakContinueBtn = document.getElementById('lessonStreakContinueBtn');
const streakCalPrevBtn = document.getElementById('streakCalPrevBtn');
const streakCalNextBtn = document.getElementById('streakCalNextBtn');
const streakCalMonthLabel = document.getElementById('streakCalMonthLabel');
const streakCalGrid = document.getElementById('streakCalGrid');

const coursesModalOverlay = document.getElementById('coursesModalOverlay');
const deleteCourseModalOverlay = document.getElementById('deleteCourseModalOverlay');
const deleteCourseModalTitle = document.getElementById('deleteCourseModalTitle');
const deleteCourseCancelBtn = document.getElementById('deleteCourseCancelBtn');
const coursesModalCloseBtn = document.getElementById('coursesModalCloseBtn');
const coursesList = document.getElementById('coursesList');
const addCourseBtn = document.getElementById('addCourseBtn');

const learnSharedCoursesBanner = document.getElementById('learnSharedCoursesBanner');
const learnSharedCoursesList = document.getElementById('learnSharedCoursesList');
const shareCourseHeaderBtn = document.getElementById('shareCourseHeaderBtn');
const shareCourseModalOverlay = document.getElementById('shareCourseModalOverlay');
const shareCourseModalSub = document.getElementById('shareCourseModalSub');
const shareCourseFriendsList = document.getElementById('shareCourseFriendsList');
const shareCourseEmpty = document.getElementById('shareCourseEmpty');
const shareCourseError = document.getElementById('shareCourseError');
const shareCourseCloseBtn = document.getElementById('shareCourseCloseBtn');
const shareCourseScanBtn = document.getElementById('shareCourseScanBtn');
const shareCourseScanStatus = document.getElementById('shareCourseScanStatus');

const createCourseModalOverlay = document.getElementById('createCourseModalOverlay');
const createCourseCancelBtn = document.getElementById('createCourseCancelBtn');
const createCourseInput = document.getElementById('createCourseInput');
const createCourseError = document.getElementById('createCourseError');
const createCourseSubmitBtn = document.getElementById('createCourseSubmitBtn');
const idkWhatToLearnBtn = document.getElementById('idkWhatToLearnBtn');

const idkQuizModalOverlay = document.getElementById('idkQuizModalOverlay');
const idkQuizBody = document.getElementById('idkQuizBody');
const idkQuizLoading = document.getElementById('idkQuizLoading');
const idkQuizProgress = document.getElementById('idkQuizProgress');
const idkQuizDots = document.getElementById('idkQuizDots');
const idkQuizQuestion = document.getElementById('idkQuizQuestion');
const idkQuizInput = document.getElementById('idkQuizInput');
const idkQuizError = document.getElementById('idkQuizError');
const idkQuizNextBtn = document.getElementById('idkQuizNextBtn');
const idkQuizCancelBtn = document.getElementById('idkQuizCancelBtn');

const lessonStartModalOverlay = document.getElementById('lessonStartModalOverlay');
const lessonStartCancelBtn = document.getElementById('lessonStartCancelBtn');
const lessonStartTitle = document.getElementById('lessonStartTitle');
const lessonStartSub = document.getElementById('lessonStartSub');
const lessonStartBtn = document.getElementById('lessonStartBtn');
const lessonStartBtnLabel = lessonStartBtn.querySelector('.lesson-start-btn-label');
const lessonStartBtnFill = lessonStartBtn.querySelector('.lesson-start-btn-fill');

const lessonViewOverlay = document.getElementById('lessonViewOverlay');
const lessonViewTitle = document.getElementById('lessonViewTitle');
const lessonExitBtn = document.getElementById('lessonExitBtn');
const lessonXpTracker = document.getElementById('lessonXpTracker');
const lessonXpTrackerCount = document.getElementById('lessonXpTrackerCount');
const xpPopupLayer = document.getElementById('xpPopupLayer');
const lessonProgressTrack = document.getElementById('lessonProgressTrack');
const lessonProgressSegments = document.getElementById('lessonProgressSegments');
const lessonSummaryXp = document.getElementById('lessonSummaryXp');
const lessonSummaryXpCount = document.getElementById('lessonSummaryXpCount');
const quitLessonModalOverlay = document.getElementById('quitLessonModalOverlay');
const quitLessonConfirmBtn = document.getElementById('quitLessonConfirmBtn');
const quitLessonCancelBtn = document.getElementById('quitLessonCancelBtn');
const lessonProgressFill = document.getElementById('lessonProgressFill');
const lessonPartContent = document.getElementById('lessonPartContent');
const lessonFeedback = document.getElementById('lessonFeedback');
const lessonWhyBtn = document.getElementById('lessonWhyBtn');
const lessonActionBtn = document.getElementById('lessonActionBtn');

// ---- AI Assistant (in-lesson chat) ----
const aiAssistantFab = document.getElementById('aiAssistantFab');
const aiAssistantPanel = document.getElementById('aiAssistantPanel');
const aiAssistantCloseBtn = document.getElementById('aiAssistantCloseBtn');
const aiAssistantMessages = document.getElementById('aiAssistantMessages');
const aiAssistantForm = document.getElementById('aiAssistantForm');
const aiAssistantInput = document.getElementById('aiAssistantInput');
const aiAssistantSendBtn = document.getElementById('aiAssistantSendBtn');

// ---- "Why?" explanation modal ----
const whyModalOverlay = document.getElementById('whyModalOverlay');
const whyModalBody = document.getElementById('whyModalBody');
const whyModalCloseBtn = document.getElementById('whyModalCloseBtn');
const lessonSummaryModalOverlay = document.getElementById('lessonSummaryModalOverlay');
const lessonSummaryContainer = document.getElementById('lessonSummaryContainer');
const lessonSummaryStreak = document.getElementById('lessonSummaryStreak');
const lessonSummaryStreakCount = document.getElementById('lessonSummaryStreakCount');
const lessonSummaryScore = document.getElementById('lessonSummaryScore');
const lessonSummaryCompleteBtn = document.getElementById('lessonSummaryCompleteBtn');

const learnGamesBtn = document.getElementById('learnGamesBtn');
const gamesPageOverlay = document.getElementById('gamesPageOverlay');
const gamesExitBtn = document.getElementById('gamesExitBtn');
const gameCardTrivia = document.getElementById('gameCardTrivia');

const triviaChooseModalOverlay = document.getElementById('triviaChooseModalOverlay');
const triviaChooseTitle = document.getElementById('triviaChooseTitle');
const triviaCourseOption = document.getElementById('triviaCourseOption');
const triviaCourseOptionDesc = document.getElementById('triviaCourseOptionDesc');
const triviaCustomOption = document.getElementById('triviaCustomOption');
const triviaCustomInput = document.getElementById('triviaCustomInput');
const triviaChooseError = document.getElementById('triviaChooseError');
const triviaGenerateBtn = document.getElementById('triviaGenerateBtn');
const triviaChooseCancelBtn = document.getElementById('triviaChooseCancelBtn');

const triviaViewOverlay = document.getElementById('triviaViewOverlay');
const triviaViewContainer = document.getElementById('triviaViewContainer');
const triviaExitBtn = document.getElementById('triviaExitBtn');
const triviaScroller = document.getElementById('triviaScroller');

// ---- Voice Trivia ----
const gameCardVoiceTrivia = document.getElementById('gameCardVoiceTrivia');
const voiceTriviaViewOverlay = document.getElementById('voiceTriviaViewOverlay');
const voiceTriviaContainer = document.getElementById('voiceTriviaContainer');
const voiceTriviaExitBtn = document.getElementById('voiceTriviaExitBtn');
const voiceTriviaProgress = document.getElementById('voiceTriviaProgress');
const voiceTriviaPlayArea = document.getElementById('voiceTriviaPlayArea');
const voiceTriviaQuestion = document.getElementById('voiceTriviaQuestion');
const voiceTriviaMic = document.getElementById('voiceTriviaMic');
const voiceTriviaStatus = document.getElementById('voiceTriviaStatus');
const voiceTriviaTranscript = document.getElementById('voiceTriviaTranscript');
const voiceTriviaFeedback = document.getElementById('voiceTriviaFeedback');
const voiceTriviaGiveUpBtn = document.getElementById('voiceTriviaGiveUpBtn');
const voiceTriviaFinished = document.getElementById('voiceTriviaFinished');
const voiceTriviaPlayAgainBtn = document.getElementById('voiceTriviaPlayAgainBtn');
const voiceTriviaExitFinishedBtn = document.getElementById('voiceTriviaExitFinishedBtn');

// ---- Maze ----
const gameCardMaze = document.getElementById('gameCardMaze');
const mazeChooseModalOverlay = document.getElementById('mazeChooseModalOverlay');
const mazeCourseOption = document.getElementById('mazeCourseOption');
const mazeCourseOptionDesc = document.getElementById('mazeCourseOptionDesc');
const mazeCustomOption = document.getElementById('mazeCustomOption');
const mazeCustomInput = document.getElementById('mazeCustomInput');
const mazeDifficultyBtns = document.querySelectorAll('.maze-difficulty-btn');
const mazeChooseError = document.getElementById('mazeChooseError');
const mazeGenerateBtn = document.getElementById('mazeGenerateBtn');
const mazeChooseCancelBtn = document.getElementById('mazeChooseCancelBtn');

const mazeViewOverlay = document.getElementById('mazeViewOverlay');
const mazeViewTitle = document.getElementById('mazeViewTitle');
const mazeExitBtn = document.getElementById('mazeExitBtn');
const mazeTimerEl = document.getElementById('mazeTimer');
const mazeGrid = document.getElementById('mazeGrid');
const mazeUpBtn = document.getElementById('mazeUpBtn');
const mazeDownBtn = document.getElementById('mazeDownBtn');
const mazeLeftBtn = document.getElementById('mazeLeftBtn');
const mazeRightBtn = document.getElementById('mazeRightBtn');

const mazeQuestionModalOverlay = document.getElementById('mazeQuestionModalOverlay');
const mazeQuestionText = document.getElementById('mazeQuestionText');
const mazeQuestionChoices = document.getElementById('mazeQuestionChoices');
const mazeQuestionFeedback = document.getElementById('mazeQuestionFeedback');

const mazeFinishedModalOverlay = document.getElementById('mazeFinishedModalOverlay');
const mazeFinishedXpCount = document.getElementById('mazeFinishedXpCount');
const mazeFinishedMoves = document.getElementById('mazeFinishedMoves');
const mazeFinishedTime = document.getElementById('mazeFinishedTime');
const mazeFinishedDoneBtn = document.getElementById('mazeFinishedDoneBtn');

// ---- Seesaw ----
const gameCardSeesaw = document.getElementById('gameCardSeesaw');
const seesawChooseModalOverlay = document.getElementById('seesawChooseModalOverlay');
const seesawCourseOption = document.getElementById('seesawCourseOption');
const seesawCourseOptionDesc = document.getElementById('seesawCourseOptionDesc');
const seesawCustomOption = document.getElementById('seesawCustomOption');
const seesawCustomInput = document.getElementById('seesawCustomInput');
const seesawDurationBtns = document.querySelectorAll('.seesaw-duration-btn');
const seesawChooseError = document.getElementById('seesawChooseError');
const seesawGenerateBtn = document.getElementById('seesawGenerateBtn');
const seesawChooseCancelBtn = document.getElementById('seesawChooseCancelBtn');

const seesawViewOverlay = document.getElementById('seesawViewOverlay');
const seesawContainer = document.getElementById('seesawContainer');
const seesawRotator = document.getElementById('seesawRotator');
const seesawFill = document.getElementById('seesawFill');
const seesawTimerMid = document.getElementById('seesawTimerMid');
const seesawPlayerTag = document.getElementById('seesawPlayerTag');
const seesawQuestionText = document.getElementById('seesawQuestionText');
const seesawChoices = document.getElementById('seesawChoices');
const seesawExitBtn = document.getElementById('seesawExitBtn');

const seesawLoseModalOverlay = document.getElementById('seesawLoseModalOverlay');
const seesawLoseIcon = document.getElementById('seesawLoseIcon');
const seesawLoseTitle = document.getElementById('seesawLoseTitle');
const seesawLoseStats = document.getElementById('seesawLoseStats');
const seesawLoseXp = document.getElementById('seesawLoseXp');
const seesawLoseDoneBtn = document.getElementById('seesawLoseDoneBtn');
const seesawWaitingBanner = document.getElementById('seesawWaitingBanner');

// ---- Seesaw: Play With a Friend (remote, cross-device via multiplayer.js) ----
const seesawModeModalOverlay = document.getElementById('seesawModeModalOverlay');
const seesawModeSameDeviceBtn = document.getElementById('seesawModeSameDeviceBtn');
const seesawModeFriendBtn = document.getElementById('seesawModeFriendBtn');
const seesawModeCancelBtn = document.getElementById('seesawModeCancelBtn');
const seesawFriendChoiceModalOverlay = document.getElementById('seesawFriendChoiceModalOverlay');
const seesawHostBtn = document.getElementById('seesawHostBtn');
const seesawJoinBtn = document.getElementById('seesawJoinBtn');
const seesawFriendChoiceCancelBtn = document.getElementById('seesawFriendChoiceCancelBtn');
const seesawHostWaitModalOverlay = document.getElementById('seesawHostWaitModalOverlay');
const seesawHostCodeDisplay = document.getElementById('seesawHostCodeDisplay');
const seesawHostQrCanvas = document.getElementById('seesawHostQrCanvas');
const seesawHostWaitStatus = document.getElementById('seesawHostWaitStatus');
const seesawHostWaitCancelBtn = document.getElementById('seesawHostWaitCancelBtn');

// ---- Duel: Play With a Friend (remote, cross-device via multiplayer.js) ----
const duelModeModalOverlay = document.getElementById('duelModeModalOverlay');
const duelModeSameDeviceBtn = document.getElementById('duelModeSameDeviceBtn');
const duelModeFriendBtn = document.getElementById('duelModeFriendBtn');
const duelModeCancelBtn = document.getElementById('duelModeCancelBtn');
const duelFriendChoiceModalOverlay = document.getElementById('duelFriendChoiceModalOverlay');
const duelHostBtn = document.getElementById('duelHostBtn');
const duelJoinBtn = document.getElementById('duelJoinBtn');
const duelFriendChoiceCancelBtn = document.getElementById('duelFriendChoiceCancelBtn');
const duelHostWaitModalOverlay = document.getElementById('duelHostWaitModalOverlay');
const duelHostCodeDisplay = document.getElementById('duelHostCodeDisplay');
const duelHostQrCanvas = document.getElementById('duelHostQrCanvas');
const duelHostWaitStatus = document.getElementById('duelHostWaitStatus');
const duelHostWaitCancelBtn = document.getElementById('duelHostWaitCancelBtn');
const joinGameModalOverlay = document.getElementById('joinGameModalOverlay');
const joinGameCodeInput = document.getElementById('joinGameCodeInput');
const joinGameError = document.getElementById('joinGameError');
const joinGameSubmitBtn = document.getElementById('joinGameSubmitBtn');
const joinGameCancelBtn = document.getElementById('joinGameCancelBtn');
const joinGameScanBtn = document.getElementById('joinGameScanBtn');

// ---- Group Game DOM refs ----
const groupGameChoiceModalOverlay = document.getElementById('groupGameChoiceModalOverlay');
const groupGameHostOption = document.getElementById('groupGameHostOption');
const groupGameJoinOption = document.getElementById('groupGameJoinOption');
const groupGameChoiceError = document.getElementById('groupGameChoiceError');
const groupGameChoiceCancelBtn = document.getElementById('groupGameChoiceCancelBtn');

const groupGameJoinModalOverlay = document.getElementById('groupGameJoinModalOverlay');
const groupGameJoinCodeInput = document.getElementById('groupGameJoinCodeInput');
const groupGameJoinError = document.getElementById('groupGameJoinError');
const groupGameJoinSubmitBtn = document.getElementById('groupGameJoinSubmitBtn');
const groupGameJoinScanBtn = document.getElementById('groupGameJoinScanBtn');
const groupGameJoinCancelBtn = document.getElementById('groupGameJoinCancelBtn');

const groupGameLobbyOverlay = document.getElementById('groupGameLobbyOverlay');
const groupGameLobbyLeaveBtn = document.getElementById('groupGameLobbyLeaveBtn');
const groupGameLobbySub = document.getElementById('groupGameLobbySub');
const groupGameLobbyCode = document.getElementById('groupGameLobbyCode');
const groupGameLobbyQrCanvas = document.getElementById('groupGameLobbyQrCanvas');
const groupGameLobbyRoster = document.getElementById('groupGameLobbyRoster');
const groupGameLobbyEmptyNote = document.getElementById('groupGameLobbyEmptyNote');
const groupGameLobbyStartBtn = document.getElementById('groupGameLobbyStartBtn');
const groupGameLobbyWaitingNote = document.getElementById('groupGameLobbyWaitingNote');

const groupGamePickerModalOverlay = document.getElementById('groupGamePickerModalOverlay');
const groupGamePicker2MinOption = document.getElementById('groupGamePicker2MinOption');
const groupGamePickerCancelBtn = document.getElementById('groupGamePickerCancelBtn');

const groupGameSetupModalOverlay = document.getElementById('groupGameSetupModalOverlay');
const groupGameSetupGameGrid = document.getElementById('groupGameSetupGameGrid');
const groupGameSetupCourseOption = document.getElementById('groupGameSetupCourseOption');
const groupGameSetupCourseOptionDesc = document.getElementById('groupGameSetupCourseOptionDesc');
const groupGameSetupCustomOption = document.getElementById('groupGameSetupCustomOption');
const groupGameSetupCustomInput = document.getElementById('groupGameSetupCustomInput');
const groupGameSetupDifficultyRow = document.getElementById('groupGameSetupDifficultyRow');
const groupGameSetupError = document.getElementById('groupGameSetupError');
const groupGameSetupStartBtn = document.getElementById('groupGameSetupStartBtn');
const groupGameSetupCancelBtn = document.getElementById('groupGameSetupCancelBtn');

const groupGameHud = document.getElementById('groupGameHud');
const groupGameHudTimer = document.getElementById('groupGameHudTimer');
const groupGameHudPlayers = document.getElementById('groupGameHudPlayers');

const groupGameRoundEndModalOverlay = document.getElementById('groupGameRoundEndModalOverlay');
const groupGameRoundEndTitle = document.getElementById('groupGameRoundEndTitle');
const groupGameRoundEndSub = document.getElementById('groupGameRoundEndSub');
const groupGameRoundEndBtn = document.getElementById('groupGameRoundEndBtn');

const groupGameResultsOverlay = document.getElementById('groupGameResultsOverlay');
const groupGameResultsList = document.getElementById('groupGameResultsList');
const groupGamePlayAgainBtn = document.getElementById('groupGamePlayAgainBtn');
const groupGameResultsWaitingNote = document.getElementById('groupGameResultsWaitingNote');
const groupGameResultsLeaveBtn = document.getElementById('groupGameResultsLeaveBtn');
const joinGameEntryBtn = document.getElementById('joinGameEntryBtn');

// ---- Duel ("Who Can Answer First?") ----
const gameCardDuel = document.getElementById('gameCardDuel');
const duelChooseModalOverlay = document.getElementById('duelChooseModalOverlay');
const duelCourseOption = document.getElementById('duelCourseOption');
const duelCourseOptionDesc = document.getElementById('duelCourseOptionDesc');
const duelCustomOption = document.getElementById('duelCustomOption');
const duelCustomInput = document.getElementById('duelCustomInput');
const duelTimerBtns = document.querySelectorAll('.duel-timer-btn');
const duelChooseError = document.getElementById('duelChooseError');
const duelGenerateBtn = document.getElementById('duelGenerateBtn');
const duelChooseCancelBtn = document.getElementById('duelChooseCancelBtn');

const duelViewOverlay = document.getElementById('duelViewOverlay');
const duelContainer = document.getElementById('duelContainer');
const duelZoneTop = document.getElementById('duelZoneTop');
const duelZoneBottom = document.getElementById('duelZoneBottom');
const duelQuestionTextTop = document.getElementById('duelQuestionTextTop');
const duelQuestionTextBottom = document.getElementById('duelQuestionTextBottom');
const duelChoicesTop = document.getElementById('duelChoicesTop');
const duelChoicesBottom = document.getElementById('duelChoicesBottom');
const duelScore1 = document.getElementById('duelScore1');
const duelScore2 = document.getElementById('duelScore2');
const duelTimerMid = document.getElementById('duelTimerMid');
const duelRoundLabel = document.getElementById('duelRoundLabel');
const duelExitBtn = document.getElementById('duelExitBtn');

const duelEndModalOverlay = document.getElementById('duelEndModalOverlay');
const duelEndIcon = document.getElementById('duelEndIcon');
const duelEndTitle = document.getElementById('duelEndTitle');
const duelEndStats = document.getElementById('duelEndStats');
const duelEndXp = document.getElementById('duelEndXp');
const duelEndDoneBtn = document.getElementById('duelEndDoneBtn');

// ---- Meltdown ----
const gameCardMeltdown = document.getElementById('gameCardMeltdown');
const meltdownChooseModalOverlay = document.getElementById('meltdownChooseModalOverlay');
const meltdownCourseOption = document.getElementById('meltdownCourseOption');
const meltdownCourseOptionDesc = document.getElementById('meltdownCourseOptionDesc');
const meltdownCustomOption = document.getElementById('meltdownCustomOption');
const meltdownCustomInput = document.getElementById('meltdownCustomInput');
const meltdownDifficultyBtns = document.querySelectorAll('.meltdown-difficulty-btn');
const meltdownChooseError = document.getElementById('meltdownChooseError');
const meltdownGenerateBtn = document.getElementById('meltdownGenerateBtn');
const meltdownChooseCancelBtn = document.getElementById('meltdownChooseCancelBtn');

const meltdownViewOverlay = document.getElementById('meltdownViewOverlay');
const meltdownExitBtn = document.getElementById('meltdownExitBtn');
const meltdownStreakEl = document.getElementById('meltdownStreak');
const meltdownTimerEl = document.getElementById('meltdownTimer');
const meltdownQuestionText = document.getElementById('meltdownQuestionText');
const meltdownChoices = document.getElementById('meltdownChoices');
const meltdownThermoFill = document.getElementById('meltdownThermoFill');

const meltdownLoseModalOverlay = document.getElementById('meltdownLoseModalOverlay');
const meltdownLoseStats = document.getElementById('meltdownLoseStats');
const meltdownLoseBest = document.getElementById('meltdownLoseBest');
const meltdownLoseXp = document.getElementById('meltdownLoseXp');
const meltdownLoseDoneBtn = document.getElementById('meltdownLoseDoneBtn');

// ---- Word Grid ----
const gameCardWordGrid = document.getElementById('gameCardWordGrid');
const wordGridChooseModalOverlay = document.getElementById('wordGridChooseModalOverlay');
const wordGridCourseOption = document.getElementById('wordGridCourseOption');
const wordGridCourseOptionDesc = document.getElementById('wordGridCourseOptionDesc');
const wordGridCustomOption = document.getElementById('wordGridCustomOption');
const wordGridCustomInput = document.getElementById('wordGridCustomInput');
const wordGridChooseError = document.getElementById('wordGridChooseError');
const wordGridGenerateBtn = document.getElementById('wordGridGenerateBtn');
const wordGridChooseCancelBtn = document.getElementById('wordGridChooseCancelBtn');

const wordGridViewOverlay = document.getElementById('wordGridViewOverlay');
const wordGridExitBtn = document.getElementById('wordGridExitBtn');
const wordGridHintText = document.getElementById('wordGridHintText');
const wordGridAttemptsLeft = document.getElementById('wordGridAttemptsLeft');
const wordGridBoard = document.getElementById('wordGridBoard');
const wordGridMessage = document.getElementById('wordGridMessage');
const wordGridKeyboard = document.getElementById('wordGridKeyboard');

const wordGridEndModalOverlay = document.getElementById('wordGridEndModalOverlay');
const wordGridEndTitle = document.getElementById('wordGridEndTitle');
const wordGridEndStats = document.getElementById('wordGridEndStats');
const wordGridEndXp = document.getElementById('wordGridEndXp');
const wordGridPlayAgainBtn = document.getElementById('wordGridPlayAgainBtn');
const wordGridEndDoneBtn = document.getElementById('wordGridEndDoneBtn');

// ---- Connectors ----
const gameCardConnectors = document.getElementById('gameCardConnectors');
const connectorsSetupOverlay = document.getElementById('connectorsSetupOverlay');
const connectorsSetupExitBtn = document.getElementById('connectorsSetupExitBtn');
const connectorsCourseList = document.getElementById('connectorsCourseList');
const connectorsCoursesEmptyNote = document.getElementById('connectorsCoursesEmptyNote');
const connectorsTopicInput = document.getElementById('connectorsTopicInput');
const connectorsAddTopicBtn = document.getElementById('connectorsAddTopicBtn');
const connectorsTopicChips = document.getElementById('connectorsTopicChips');
const connectorsDifficultyBtns = document.querySelectorAll('.connectors-difficulty-btn');
const connectorsDifficultyHint = document.getElementById('connectorsDifficultyHint');
const connectorsSetupError = document.getElementById('connectorsSetupError');
const connectorsGenerateBtn = document.getElementById('connectorsGenerateBtn');

const connectorsViewOverlay = document.getElementById('connectorsViewOverlay');
const connectorsExitBtn = document.getElementById('connectorsExitBtn');
const connectorsShuffleBtn = document.getElementById('connectorsShuffleBtn');
const connectorsSolvedBands = document.getElementById('connectorsSolvedBands');
const connectorsGrid = document.getElementById('connectorsGrid');
const connectorsFeedback = document.getElementById('connectorsFeedback');
const connectorsMistakesDots = document.getElementById('connectorsMistakesDots');
const connectorsDeselectBtn = document.getElementById('connectorsDeselectBtn');
const connectorsSubmitBtn = document.getElementById('connectorsSubmitBtn');
const connectorsActionsRow = document.getElementById('connectorsActionsRow');
const connectorsContinueBtn = document.getElementById('connectorsContinueBtn');

const connectorsEndModalOverlay = document.getElementById('connectorsEndModalOverlay');
const connectorsEndIcon = document.getElementById('connectorsEndIcon');
const connectorsEndTitle = document.getElementById('connectorsEndTitle');
const connectorsEndStats = document.getElementById('connectorsEndStats');
const connectorsEndXp = document.getElementById('connectorsEndXp');
const connectorsEndDoneBtn = document.getElementById('connectorsEndDoneBtn');

const learnNavBtn = document.querySelector('.nav-btn[data-page="learn"]');

// ---- State ----
let courses = [];              // all of the user's courses (metadata only)
let activeCourse = null;       // full course doc + id, currently shown on course home
let pendingShares = [];        // incoming course shares awaiting Accept/Decline
let unsubscribeSharedCourses = null; // live listener handle for pendingShares
let shareCourseTarget = null;  // the course currently open in the Share Course modal
let viewedUnitIndex = null;    // which unit's lesson path is currently open (null = units overview)
let learnProfile = { streak: 0, xp: 0, lastLessonDate: null, missedDaysInRow: 0, completedDates: [] };

// shop.js owns a live onSnapshot listener on the shop-related learnProfile
// fields, so a purchase made in the XP Shop reflects here immediately
// instead of only after the next loadStreak() (e.g. app reload). Merge
// onto learnProfile rather than replacing it wholesale, since learnProfile
// also holds streak/xp/completedDates fields shop.js doesn't own.
onShopStateChange((shopFields) => {
  if (!learnProfile) return;
  learnProfile.streakPassCount = shopFields.streakPassCount;
  learnProfile.passDates = shopFields.passDates;
  learnProfile.unlockedColors = shopFields.unlockedColors;
  learnProfile.unlockedEmoji = shopFields.unlockedEmoji;
  learnProfile.purchasedReviewCredits = shopFields.purchasedReviewCredits;
  learnProfile.bankedPremiumDays = shopFields.bankedPremiumDays;
  learnProfile.extraCourseSlotBought = shopFields.extraCourseSlotBought;
  // onShopStateChange fires synchronously on registration (with whatever's
  // cached), which happens here at module-eval time — before later
  // `let`/`const` declarations further down this file (wrongAnswers,
  // streakModalOverlay's DOM ties, etc) have run. Defer the UI-touching
  // part to a microtask so it never executes before the rest of the
  // module has finished initializing.
  queueMicrotask(() => {
    if (typeof updateReviewWrongAnswersBtn === 'function') updateReviewWrongAnswersBtn();
    if (streakModalOverlay?.classList.contains('show') && streakPassCountLabel) {
      const passCount = learnProfile.streakPassCount || 0;
      streakPassCountLabel.textContent = passCount;
      streakPassPlural.textContent = passCount === 1 ? '' : 'es';
    }
  });
});
let calendarViewDate = new Date();
let pendingLessonRef = null;   // { unitIndex, lessonIndex } chosen from path, shown in start modal
let activeLessonGenToken = null; // { cancelled } for the in-flight lesson generation, if any — see lessonStartBtn handler
let titleFetchPromises = new Map(); // `${unitIndex}_${lessonIndex}` -> in-flight title pre-generation promise
let currentLesson = null;      // { id, ref, parts, isCourseReview, ... } currently open in lesson view
let wrongAnswers = [];         // wrong answers stored in Firestore for the active course
let wrongAnswersCourseId = null; // which course `wrongAnswers` was last loaded for
let _consumingPurchasedReview = false; // true if the in-progress review is spending an XP Shop credit

// ---- Review Page: AI-picked weak spots / strengths (learnProfile.weakSpots / .strengths) ----
// Distinct from the legacy wrongAnswers bank above (which now exists ONLY
// to feed Daily/Combo Lesson's mistake-mixing — see startComboLesson).
// { id, title, description, exampleQuestion, courseId, createdAt } — capped
// at 5 each, newest first, oldest dropped once a 6th is added.
let lessonAnswerLog = [];      // [{ question, choices, correctIndex, selectedIndex }, ...] this lesson, for weak/strong-spot generation
const REVIEW_LESSONS_ENABLED_KEY = 'kll_review_lessons_enabled';
function getReviewLessonsEnabled() {
  try {
    const stored = localStorage.getItem(REVIEW_LESSONS_ENABLED_KEY);
    return stored === null ? true : stored === '1';
  } catch { return true; }
}
const REVIEW_FREE_USES_PER_DAY = 3;
function reviewFreeUsesKey() {
  return `kll_review_free_uses_${uid()}`;
}
// Local-midnight-reset counter for the 3-free-reviews/day cap (free
// accounts only — premium/banked-premium-day is unlimited). Stored as
// { date: 'YYYY-MM-DD', count: n } in localStorage, same reset scheme as
// shop.js's todayStrLocal()-keyed date fields.
function getReviewFreeUsesToday() {
  try {
    const raw = localStorage.getItem(reviewFreeUsesKey());
    const parsed = raw ? JSON.parse(raw) : null;
    if (!parsed || parsed.date !== shopTodayStr()) return 0;
    return parsed.count || 0;
  } catch { return 0; }
}
function bumpReviewFreeUsesToday() {
  try {
    const count = getReviewFreeUsesToday() + 1;
    localStorage.setItem(reviewFreeUsesKey(), JSON.stringify({ date: shopTodayStr(), count }));
  } catch {}
}
function hasFreeReviewUseLeft() {
  return getReviewFreeUsesToday() < REVIEW_FREE_USES_PER_DAY;
}
// Daily Lesson (formerly Combo Lesson) — "done today" is saved locally
// only, per spec, greying the button out until local midnight.
function dailyLessonDoneKey() {
  return `kll_daily_lesson_done_${uid()}`;
}
function getDailyLessonDoneToday() {
  try { return localStorage.getItem(dailyLessonDoneKey()) === shopTodayStr(); } catch { return false; }
}
function setDailyLessonDoneToday() {
  try { localStorage.setItem(dailyLessonDoneKey(), shopTodayStr()); } catch {}
}

// ---- Group Game state ----
let groupGameCode = null;          // current room's 4-char code, null when not in a room
let groupGameRole = null;          // 'host' | 'guest'
let groupGameUnsubscribe = null;   // listenToRoom() unsubscribe handle
let groupGameLatestRoom = null;    // most recent room snapshot from the listener
let groupGameActive = false;       // true while a "2 Minutes" round is actually playing on this device
let groupGamePendingGameId = null; // game picked in the setup modal, before Start Round is pressed
let groupGameGameId = null;        // game running in the current active round
let groupGameTopic = '';
let groupGameDifficulty = 'medium';
let groupGameCountdownHandle = null;
let groupGameRoundEndsAt = 0;
let groupGameResultsShown = false; // guards against double-showing the results screen

function myGroupGamePlayerName() {
  return (auth.currentUser?.displayName || auth.currentUser?.email?.split('@')[0] || 'Player').split(' ')[0];
}
let currentPartIndex = 0;
let selectedChoiceIndex = null;
let answerLocked = false;
let lessonQuestionCount = 0;
let lessonCorrectCount = 0;
let lessonXp = 10;             // this lesson's running XP total (starts at 10, only persisted at completion)
// ---- Lesson-complete flow sequencing (XP overlay -> streak overlay -> badge celebrations -> home) ----
let pendingStreakExtended = false; // whether this completion should show the streak overlay next
let streakModalPostLessonFlow = false; // true while streakModalOverlay is open as part of that sequence, not the manual "view my streak" open
let lessonStreakCount = 0;     // consecutive correct answers within this lesson
let lessonAnsweredAny = false; // true once the first question has been checked — gates the quit-confirm modal
let hasInitialized = false;

// ---- AI Assistant state ----
let aiAssistantHistory = [];   // [{ role: 'user'|'assistant', content }, ...] for the current lesson
let aiAssistantBusy = false;   // true while waiting on a reply, to prevent double-sends

// ---- "Why?" state: the most recently missed question, kept around so the
// Why modal has something to explain when tapped ----
let missedQuestion = null;     // { question, choices, correctIndex, selectedIndex }

// ---- Trivia state ----
let triviaQuestions = [];   // [{ question, answer }, ...] currently loaded set
let triviaReady = false;    // true once a set has been generated and Start is available
let triviaLastTopic = null; // topic string used for the current set, reused by "Generate More"

// ---- Voice Trivia state ----
// The choose-modal (topic/course picker + "Generate") is shared with regular
// Trivia — triviaVoiceMode just decides what happens once a set is ready:
// startTriviaView() (scroll feed) or startVoiceTriviaView() (voice game).
let triviaVoiceMode = false;
let vtIndex = 0;                  // index into triviaQuestions for the current question
let vtSessionId = 0;               // bumped on every exit/restart so stray async callbacks from a previous round no-op
let vtMicStream = null;            // getUserMedia MediaStream, kept alive across questions to avoid re-prompting
let vtMimeInfo = null;             // { mime, ext } — best supported recording format, picked once
let vtMediaRecorder = null;        // current in-flight MediaRecorder
let vtRecordingLoopActive = false; // true while the record-5s/transcribe/repeat loop should keep going

// ---- Maze state ----
const MAZE_DIFFICULTY = {
  easy:   { size: 6,  interval: 15 },
  medium: { size: 8,  interval: 10 },
  hard:   { size: 10, interval: 7 },
};
let mazeQuestions = [];        // [{ question, choices, correctIndex }, ...] fetched set
let mazeReady = false;         // true once a set has been generated and Generate becomes Start
let mazeChosenDifficulty = 'medium';
let mazeGridCells = null;      // 2D array of { r, c, walls: {top,right,bottom,left} }
let mazeSize = 0;
let mazePath = [];             // history of {r,c} visited, current position = last entry
let mazeQuestionIndex = 0;     // pointer into a shuffled mazeQuestions, wraps around
let mazeTimerHandle = null;
let mazeSecondsLeft = 0;
let mazeIntervalSeconds = 0;
let mazeXp = 120;              // starts at 120, -1 per move, floor of 15
let mazeStartTime = 0;         // Date.now() when the maze run started, for the complete screen's time-taken stat
let mazeAwaitingAnswer = false; // true while the blocking question modal is up
let mazeActive = false;         // true once the maze view is open and playable

// ---- Seesaw state ----
// duration in seconds per turn, or null for infinite (no timer / no lose condition)
const SEESAW_DURATIONS = { '10': 10, '20': 20, '30': 30, infinite: null };
let seesawQuestions = [];        // [{ question, choices, correctIndex }, ...] fetched set
let seesawReady = false;         // true once a set has been generated and Generate becomes Start
let seesawChosenDuration = '10';
let seesawLastTopic = null;      // topic string reused when fetching more questions mid-game
let seesawUsedQuestions = [];    // question texts already served, sent back so regenerated sets don't repeat
let seesawHistoryContext = [];   // cross-session history (gameHistory.js), fetched once per game and reused for mid-game top-ups
let seesawFetchingMore = false;  // true while a background top-up fetch is in flight
let seesawIndex = 0;             // pointer into seesawQuestions
let seesawCurrentQuestion = null;
let seesawCurrentPlayer = 1;     // 1 or 2 — whoever is currently answering
let seesawDurationSeconds = 10;  // resolved seconds for the current game, or null if infinite
let seesawStartTime = 0;
let seesawGameStartTime = 0;     // Date.now() when the whole game began (not per-turn), for XP
const SEESAW_WINNER_XP_BONUS = 30; // extra XP the winner gets over the loser in a remote match
const DUEL_WINNER_XP_BONUS = 30;   // extra XP the winner gets over the loser in a remote match
let seesawTickHandle = null;
let seesawLoseTimeoutHandle = null;
let seesawActive = false;        // true once the seesaw view is open and playable

// ---- Seesaw: Play With a Friend (remote) state ----
const SEESAW_MIN_DURATION = 3;   // floor the turn timer speeds down to, in seconds
let seesawMode = 'local';        // 'local' (pass-and-play) | 'host' | 'guest' (remote via multiplayer.js)
let seesawSessionCode = null;    // join code for the active remote session, if any
let seesawMyPlayer = 1;          // which player number (1 or 2) this device is, in remote mode
let seesawUnsubscribe = null;    // cleanup fn for the active RTDB session listener
let seesawLatestSession = null;  // most recent RTDB session payload seen, even before the view is open
let seesawBaseDurationSeconds = 10; // immutable per-game base the speed-up counts down from
let seesawCorrectCount = 0;      // total correct answers this game (both players) — drives the speed-up
let seesawScores = { 1: 0, 2: 0 };
let seesawLastRemoteQuestionIndex = -1;
let seesawLastRemoteTurn = null;

// ---- Duel ("Who Can Answer First?") state ----
const DUEL_TIMERS = { '5': 5, '8': 8, '12': 12 };
const DUEL_TOTAL_ROUNDS = 10;
let duelQuestions = [];          // [{ question, choices, correctIndex }, ...] fetched set
let duelReady = false;           // true once a set has been generated and Generate becomes Start
let duelChosenTimer = '8';
let duelDurationSeconds = 8;
let duelLastTopic = null;        // topic string reused when fetching more questions mid-match
let duelUsedQuestions = [];      // question texts already served
let duelHistoryContext = [];     // cross-session history (gameHistory.js), fetched once per game and reused for mid-game top-ups
let duelFetchingMore = false;    // true while a background top-up fetch is in flight
let duelIndex = 0;               // pointer into duelQuestions
let duelCurrentQuestion = null;
let duelRoundNumber = 1;         // 1-based, shown as "Round X of 10"
let duelScores = { 1: 0, 2: 0 };
let duelStartTime = 0;
let duelTickHandle = null;
let duelRoundTimeoutHandle = null;
let duelActive = false;          // true once the duel view is open and playable

// ---- Duel: Play With a Friend (remote) state ----
let duelMode = 'local';          // 'local' (same device) | 'host' | 'guest' (remote via multiplayer.js)
let duelSessionCode = null;      // join code for the active remote session, if any
let duelMyPlayer = 1;            // which player number (1 or 2) this device is, in remote mode
let duelUnsubscribe = null;      // cleanup fn for the active RTDB session listener
let duelLatestSession = null;    // most recent RTDB session payload seen, even before the view is open
let duelLastRemoteRoundIndex = -1;
let duelRemoteTotalRounds = DUEL_TOTAL_ROUNDS;
let duelRoundResolved = false;   // true once this round has a winner/timeout, ignore further taps

// ---- Meltdown state ----
// "Starting Heat" difficulty just sets where the lava bar begins — the real
// difficulty driver is the per-question timer, which always starts at
// MELTDOWN_START_SECONDS and shaves MELTDOWN_SECONDS_STEP off every question.
const MELTDOWN_DIFFICULTY = { easy: 10, medium: 25, hard: 40 }; // starting heat %
const MELTDOWN_START_SECONDS = 10;
const MELTDOWN_SECONDS_STEP = 0.5;
const MELTDOWN_MIN_SECONDS = 3;
const MELTDOWN_HEAT_MAX = 100;
const MELTDOWN_COOL_PER_CORRECT = 12; // heat removed on a correct answer
const MELTDOWN_HEAT_PER_WRONG = 10;   // heat added on a wrong or timed-out answer
const MELTDOWN_PASSIVE_HEAT_PER_SECOND = 1.4; // heat that creeps up every second, just from the clock running
const MELTDOWN_BEST_KEY = 'meltdownBestStreak';

let meltdownQuestions = [];        // [{ question, choices, correctIndex }, ...] fetched set
let meltdownReady = false;         // true once a set has been generated and Generate becomes Start
let meltdownChosenDifficulty = 'medium';
let meltdownIndex = 0;             // pointer into a shuffled meltdownQuestions, wraps around
let meltdownCurrentQuestion = null;
let meltdownHeat = 0;              // 0-100, melts at MELTDOWN_HEAT_MAX
let meltdownSecondsForQuestion = MELTDOWN_START_SECONDS; // current question's total time budget
let meltdownQuestionsAnswered = 0; // drives the per-question shrink
let meltdownStreak = 0;            // correct answers in a row this run
let meltdownTickHandle = null;
let meltdownTimeoutHandle = null;
let meltdownStartTime = 0;
let meltdownGameStartTime = 0;     // Date.now() when the whole run began (not per-question), for XP
let meltdownActive = false;        // true once the meltdown view is open and playable

// ---- Word Grid state ----
const WORDGRID_WORD_LENGTH = 5;
const WORDGRID_MAX_GUESSES = 6;
// XP for a win, indexed by (guesses used - 1) — fewer guesses earns more.
const WORDGRID_XP_BY_GUESS = [100, 85, 70, 55, 40, 25];
const WORDGRID_LOSE_XP = 15; // small consolation XP for using all 6 guesses
const WORDGRID_KEY_ROWS = [
  ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
  ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
  ['ENTER', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'DEL'],
];

let wordGridWord = '';             // secret word, uppercase
let wordGridHint = '';             // short topic-relevant hint from the model
let wordGridTopic = '';            // topic string, reused by "Play Again"
let wordGridReady = false;         // true once a word has been generated and Generate becomes Start
let wordGridGuesses = [];          // committed guesses so far, e.g. ["CRANE", ...]
let wordGridCurrentGuess = '';     // letters typed for the in-progress row
let wordGridActive = false;        // true once the view is open and playable
let wordGridKeyStates = {};        // { A: 'correct'|'present'|'absent' } best-known state per letter

// ---- Connectors state ----
// Board scales with how many courses/topics the player picks — one group of
// 4 words per topic — rather than the fixed 4x4 NYT format.
const CONNECTORS_MIN_TOPICS = 1;
const CONNECTORS_MAX_TOPICS = 2;   // 1 topic is the sweet spot — the AI invents extra angles itself; keep it focused rather than diluted across many topics
const CONNECTORS_MISTAKES_BY_DIFFICULTY = { easy: 5, medium: 4, hard: 3 };
const CONNECTORS_DIFFICULTY_HINTS = {
  easy: 'Clearer groupings, few sneaky overlaps.',
  medium: 'A fair challenge — some words could fit more than one group.',
  hard: 'Expect real overlap and misdirection.',
};
// Fallback palette for custom (non-course) topics, which have no course color.
const CONNECTORS_FALLBACK_COLORS = ['#8B5CF6', '#F97316', '#0EA5E9', '#EC4899', '#10B981', '#F59E0B'];
const CONNECTORS_XP_SOLVED_BASE = 60;
const CONNECTORS_XP_MISTAKE_PENALTY = 8;
const CONNECTORS_XP_SOLVED_MIN = 20;
const CONNECTORS_XP_INCOMPLETE_BASE = 15;
const CONNECTORS_XP_PER_GROUP_SOLVED = 5;   // consolation XP per group solved before a loss

let connectorsSelectedCourseIds = new Set(); // course ids checked in the setup page
let connectorsCustomTopics = [];    // extra free-text topics added into the mix
let connectorsChosenDifficulty = 'medium';
let connectorsGroups = [];          // [{ id, label, color, words: [4 strings], solved }, ...]
let connectorsTileMap = new Map();  // tile key ("g0::WORD") -> { key, word, groupId }
let connectorsDisplayOrder = [];    // keys of tiles currently on the board (unsolved), in display order
let connectorsSelected = [];        // tile keys currently tapped (max 4)
let connectorsGuessedSets = [];     // previously-submitted 4-key combos, sorted, to block exact repeats
let connectorsMaxMistakes = 4;
let connectorsMistakesLeft = 4;

// ============================================================
// INIT — runs once, the first time the Learn tab is opened
// ============================================================
if (learnNavBtn) {
  learnNavBtn.addEventListener('click', () => {
    if (!hasInitialized) {
      hasInitialized = true;
      initLearn();
    }
  });
}

// ---- Learn cache: same idea as Home's (see refreshHome/homeCacheKey in
// main.js) — paint instantly from whatever this device last saw so the
// skeleton only ever shows the very first time there's nothing cached,
// while loadStreak()/loadCourses() below still always run for real and
// repaint with fresh data. Keyed by uid.
function learnCacheKey(u) { return `kll_learn_cache_${u}`; }
function readLearnCache(u) {
  try {
    const raw = localStorage.getItem(learnCacheKey(u));
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}
function writeLearnCache(u) {
  try {
    localStorage.setItem(learnCacheKey(u), JSON.stringify({ learnProfile, courses, activeCourse }));
  } catch { /* storage full/unavailable — caching is a nicety, not required */ }
}

async function initLearn() {
  const u = uid();
  const cached = u ? readLearnCache(u) : null;
  if (cached) {
    // Paint immediately so returning to Learn never re-shows the skeleton
    // once this device has loaded it at least once.
    learnProfile = cached.learnProfile;
    courses = cached.courses;
    activeCourse = cached.activeCourse;
    learnStreakCount.textContent = learnProfile?.streak ?? 0;
    renderCourseHomeOrEmpty();
  }
  await loadStreak();
  await loadCourses();
  renderCourseHomeOrEmpty();
  startSharedCoursesListener();
  if (u) writeLearnCache(u);
  // First-ever Learn open, per account — fire-and-forget so it never
  // blocks the actual Learn page from rendering.
  maybeShowOverlay('adaptiveDifficultyWelcome');
}

// Called from main.js whenever the signed-in user changes (sign in / switch
// account / sign out), so that stale in-memory state from the previous
// account doesn't leak into the newly signed-in account's Learn tab.
// Lets other modules (e.g. the Home page) make sure courses/streak/wrong
// answers are loaded before they open a Learn-owned modal (streak, games,
// wrong-answer review) — mirrors the lazy init that normally only happens
// the first time the learner taps the Learn tab themselves.
export async function ensureLearnInitialized() {
  if (!hasInitialized) {
    hasInitialized = true;
    await initLearn();
  }
}

// Lets other modules (e.g. the episode modal's "Create a Course" button)
// open the Create Course modal pre-filled with a topic, making sure Learn
// state is loaded first even if the learner hasn't visited the Learn tab
// yet this session.
export async function openCreateCourseModalWithTopic(topic) {
  await ensureLearnInitialized();
  openCreateCourseModal(topic);
}

// Lets other modules (the episode modal's "Generate Trivia" button) skip
// straight to a 20-question trivia set on a given topic — no Trivia Choose
// modal, same worker call/gating/history plumbing as picking "Custom topic"
// there would. opts.onStart fires right before the trivia view opens (so
// the caller can dismiss its own modal at the right moment); opts.onError
// fires with a message if generation fails, and nothing else happens.
export async function generateTriviaFromTopic(topic, opts = {}) {
  const { onStart, onError } = opts;
  await ensureLearnInitialized();
  triviaVoiceMode = false;
  try {
    const history = await getGameHistoryContext('trivia');
    const data = await fetchTrivia(topic, history);
    triviaQuestions = data.questions;
    triviaLastTopic = topic;
    onStart?.();
    gateAndStartGame(startTriviaView, 'trivia', triviaQuestions.map((q) => q.question));
  } catch (err) {
    onError?.(err.message || 'Could not create trivia. Try again.');
  }
}

// Lets other modules (currently just Home, in main.js) read the exact same
// streak numbers driving Learn's own top-left counter (#learnStreakCount)
// instead of recomputing streak decay a second time and risking the two
// displays drifting apart. Always call ensureLearnInitialized() first so
// this is never read before loadStreak()'s decay logic has actually run at
// least once this session — after that, learnProfile here is kept live by
// applyStreakDecay()/bumpStreak(), so every call just reflects whatever
// Learn's own counter currently shows.
export function getLearnStreakSnapshot() {
  return {
    streak: learnProfile?.streak ?? 0,
    missedDaysInRow: learnProfile?.missedDaysInRow ?? 0,
    lastLessonDate: learnProfile?.lastLessonDate ?? null,
  };
}

// Lets Home (main.js) show a weak-spot count on its stat card without
// duplicating learn.js's own learnProfile state.
export function getWeakSpotCount() {
  return (learnProfile?.weakSpots || []).length;
}

export function resetLearnState() {
  hasInitialized = false;
  courses = [];
  activeCourse = null;
  learnProfile = { streak: 0, xp: 0, lastLessonDate: null, missedDaysInRow: 0, completedDates: [], weakSpots: [], strengths: [] };
  if (learnStreakCount) learnStreakCount.textContent = '0';
  stopSharedCoursesListener();
  pendingShares = [];
  renderSharedCoursesBanner();
}

function uid() {
  return auth.currentUser?.uid || null;
}

function escapeHtml(str) {
  return (str || '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// Lesson parts can carry EITHER a named "template"+"templateData" object
// (preferred — rendered below as real HTML/CSS by renderDiagramTemplate())
// OR a rare raw AI-generated inline SVG diagram (see learn-worker's
// generateLesson). The worker prompt constrains what either should look
// like, but this is a second, defensive layer before anything generated
// gets dropped into innerHTML: strip script-bearing/event-handler content
// and require raw svg to actually look like a well-formed <svg>. Returns
// '' (falsy) for anything that doesn't pass, so callers can just do
// `if (safeSvg) ...`.
function sanitizeLessonSvg(raw) {
  if (!raw || typeof raw !== 'string') return '';
  let svg = raw.trim();
  if (!/^<svg[\s>]/i.test(svg) || !/<\/svg>\s*$/i.test(svg)) return '';
  if (svg.length > 4000) return ''; // guard against a runaway/malformed generation
  if (/<script|<foreignObject|<iframe|javascript:/i.test(svg)) return '';
  // Strip any on*="..." / on*='...' event-handler attributes.
  svg = svg.replace(/\son[a-z]+\s*=\s*"[^"]*"/gi, '').replace(/\son[a-z]+\s*=\s*'[^']*'/gi, '');
  return svg;
}

// Lesson parts can also carry a real photo (Pexels imageUrl + imageAlt +
// imageCredit — see the worker's Pexels option in generateLessonDiagrams)
// instead of a template/svg diagram, for content where an actual photo
// teaches better than an illustration (a specific animal, landmark, food,
// etc). The worker already validates this server-side, but same as
// sanitizeLessonSvg above, this is a second defensive layer before it's
// ever dropped into an <img src>: only accept a well-formed https URL
// actually hosted on Pexels' own image CDN. Returns '' (falsy) for
// anything that doesn't pass.
function sanitizeLessonImageUrl(raw) {
  if (!raw || typeof raw !== 'string') return '';
  try {
    const u = new URL(raw.trim());
    if (u.protocol !== 'https:') return '';
    if (!/(^|\.)images\.pexels\.com$/.test(u.hostname)) return '';
    return u.href;
  } catch {
    return '';
  }
}

// ============================================================
// DIAGRAM TEMPLATES — client-side HTML/CSS renderers.
//
// Lesson parts can carry EITHER a named "template" + "templateData" object
// (preferred — rendered as real HTML/CSS below, styled in index.html under
// ".diagram-*") OR a rare raw "svg" fallback (see sanitizeLessonSvg above).
// Each renderer here mirrors the defensive "Array.isArray(...) ? ... : []"
// style already used elsewhere in this file, and escapes all AI-provided
// text via escapeHtml() before it goes into innerHTML — the worker already
// validated types/sizes, but text content still gets escaped here since
// that's what actually prevents HTML injection.
// ============================================================

const DIAGRAM_ACCENT = { accent1: 'var(--diagram-accent1)', accent2: 'var(--diagram-accent2)', none: 'transparent' };

function clampNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function renderDiagramChess(data = {}) {
  const pieces = Array.isArray(data.pieces) ? data.pieces : [];
  const highlights = Array.isArray(data.highlights) ? data.highlights : [];
  const arrow = data.arrow && data.arrow.from && data.arrow.to ? data.arrow : null;
  const glyphs = { K: '♔', Q: '♕', R: '♖', B: '♗', N: '♘', P: '♙' };
  const sqRe = /^[a-h][1-8]$/;

  const hiSet = new Set(highlights.filter((s) => sqRe.test(s)));
  const pieceMap = new Map();
  for (const p of pieces) {
    if (p && sqRe.test(p.square)) pieceMap.set(p.square, p);
  }

  let squares = '';
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const file = 'abcdefgh'[f];
      const rank = 8 - r;
      const sq = `${file}${rank}`;
      const isLight = (r + f) % 2 === 0;
      const isHi = hiSet.has(sq);
      const piece = pieceMap.get(sq);
      let inner = '';
      if (piece) {
        const glyph = glyphs[String(piece.piece || '').toUpperCase()];
        if (glyph) {
          inner = `<span class="diagram-chess-piece diagram-chess-piece-${piece.color === 'b' ? 'b' : 'w'}">${glyph}</span>`;
        }
      }
      squares += `<div class="diagram-chess-sq ${isLight ? 'light' : 'dark'} ${isHi ? 'hi' : ''}" data-sq="${sq}">${inner}</div>`;
    }
  }

  let arrowEl = '';
  if (arrow && sqRe.test(arrow.from) && sqRe.test(arrow.to)) {
    const fx = arrow.from.charCodeAt(0) - 97, fy = 8 - parseInt(arrow.from[1], 10);
    const tx = arrow.to.charCodeAt(0) - 97, ty = 8 - parseInt(arrow.to[1], 10);
    const x1 = (fx + 0.5) / 8 * 100, y1 = (fy + 0.5) / 8 * 100;
    const x2 = (tx + 0.5) / 8 * 100, y2 = (ty + 0.5) / 8 * 100;
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    const angle = Math.atan2(dy, dx) * 180 / Math.PI;
    arrowEl = `<div class="diagram-chess-arrow" style="left:${x1}%; top:${y1}%; width:${len}%; transform: rotate(${angle}deg);"></div>`;
  }

  const files = 'abcdefgh'.split('').map((f) => `<span>${f}</span>`).join('');
  const ranks = [8, 7, 6, 5, 4, 3, 2, 1].map((r) => `<span>${r}</span>`).join('');

  return `<div class="diagram-chess-wrap">
    <div class="diagram-chess-ranks">${ranks}</div>
    <div class="diagram-chess-board-col">
      <div class="diagram-chess-board">${squares}${arrowEl}</div>
      <div class="diagram-chess-files">${files}</div>
    </div>
  </div>`;
}

function renderDiagramGraph(data = {}) {
  const xMin = clampNum(data.xMin, -10), xMax = clampNum(data.xMax, 10);
  const yMin = clampNum(data.yMin, -10), yMax = clampNum(data.yMax, 10);
  const points = Array.isArray(data.points) ? data.points : [];
  const lines = Array.isArray(data.lines) ? data.lines : [];
  const spanX = (xMax - xMin) || 1, spanY = (yMax - yMin) || 1;
  const px = (x) => ((clampNum(x) - xMin) / spanX) * 100;
  const py = (y) => 100 - ((clampNum(y) - yMin) / spanY) * 100;

  const zeroX = px(0), zeroY = py(0);
  const axesHtml = `
    ${xMin <= 0 && xMax >= 0 ? `<div class="diagram-graph-axis-v" style="left:${zeroX}%;"></div>` : ''}
    ${yMin <= 0 && yMax >= 0 ? `<div class="diagram-graph-axis-h" style="top:${zeroY}%;"></div>` : ''}
  `;

  const colors = ['var(--diagram-accent1)', 'var(--diagram-accent2)'];
  let linesHtml = '';
  lines.forEach((ln, i) => {
    const pts = Array.isArray(ln.points) ? ln.points : [];
    for (let j = 0; j < pts.length - 1; j++) {
      const [x1, y1] = pts[j], [x2, y2] = pts[j + 1];
      const X1 = px(x1), Y1 = py(y1), X2 = px(x2), Y2 = py(y2);
      const len = Math.hypot(X2 - X1, Y2 - Y1);
      const angle = Math.atan2(Y2 - Y1, X2 - X1) * 180 / Math.PI;
      linesHtml += `<div class="diagram-graph-line" style="left:${X1}%; top:${Y1}%; width:${len}%; transform: rotate(${angle}deg); background: ${colors[i % 2]};"></div>`;
    }
  });

  const ptsHtml = points.map((p) => `
    <div class="diagram-graph-point" style="left:${px(p.x)}%; top:${py(p.y)}%;">
      ${p.label ? `<span class="diagram-graph-point-label">${escapeHtml(p.label)}</span>` : ''}
    </div>`).join('');

  return `<div class="diagram-graph-wrap">
    <div class="diagram-graph-grid">${axesHtml}${linesHtml}${ptsHtml}</div>
    <div class="diagram-graph-labels">
      ${data.xLabel ? `<span class="diagram-graph-xlabel">${escapeHtml(data.xLabel)}</span>` : '<span></span>'}
      ${data.yLabel ? `<span class="diagram-graph-ylabel">${escapeHtml(data.yLabel)}</span>` : ''}
    </div>
  </div>`;
}

function renderDiagramShapes(data = {}) {
  const shapes = Array.isArray(data.shapes) ? data.shapes : [];
  let body = '';
  for (const s of shapes) {
    const fillVar = DIAGRAM_ACCENT[s.fill] === DIAGRAM_ACCENT.none ? 'transparent' : DIAGRAM_ACCENT[s.fill] || 'transparent';
    if (s.type === 'circle' && typeof s.cx === 'number') {
      const r = clampNum(s.r, 40);
      const left = ((s.cx - r) / 300) * 100, top = ((s.cy - r) / 300) * 100;
      const size = (r * 2 / 300) * 100;
      body += `<div class="diagram-shape diagram-shape-circle" style="left:${left}%; top:${top}%; width:${size}%; height:${size}%; background:${fillVar};">
        ${s.label ? `<span class="diagram-shape-label">${escapeHtml(s.label)}</span>` : ''}
      </div>`;
    } else if (Array.isArray(s.points) && s.points.length >= 3) {
      const xs = s.points.map((p) => p[0]), ys = s.points.map((p) => p[1]);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const w = Math.max(1, maxX - minX), h = Math.max(1, maxY - minY);
      const clip = s.points.map((p) => `${((p[0] - minX) / w) * 100}% ${((p[1] - minY) / h) * 100}%`).join(', ');
      const cx = xs.reduce((a, b) => a + b, 0) / xs.length, cy = ys.reduce((a, b) => a + b, 0) / ys.length;
      body += `<div class="diagram-shape diagram-shape-poly" style="left:${(minX / 300) * 100}%; top:${(minY / 300) * 100}%; width:${(w / 300) * 100}%; height:${(h / 300) * 100}%; clip-path: polygon(${clip}); background:${fillVar === 'transparent' ? 'var(--diagram-poly-fill)' : fillVar};"></div>`;
      if (s.label) {
        body += `<span class="diagram-shape-freelabel" style="left:${(cx / 300) * 100}%; top:${(cy / 300) * 100}%;">${escapeHtml(s.label)}</span>`;
      }
    } else if (s.type === 'rectangle' && Array.isArray(s.points) && s.points.length === 2) {
      const [[x1, y1], [x2, y2]] = s.points;
      const left = Math.min(x1, x2), top = Math.min(y1, y2);
      const w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
      body += `<div class="diagram-shape diagram-shape-rect" style="left:${(left / 300) * 100}%; top:${(top / 300) * 100}%; width:${(w / 300) * 100}%; height:${(h / 300) * 100}%; background:${fillVar};">
        ${s.label ? `<span class="diagram-shape-label">${escapeHtml(s.label)}</span>` : ''}
      </div>`;
    }
  }
  for (const a of (data.angleLabels || [])) {
    body += `<span class="diagram-shape-tag diagram-shape-tag-accent" style="left:${(a.x / 300) * 100}%; top:${(a.y / 300) * 100}%;">${escapeHtml(a.text)}</span>`;
  }
  for (const sl of (data.sideLabels || [])) {
    body += `<span class="diagram-shape-tag" style="left:${(sl.x / 300) * 100}%; top:${(sl.y / 300) * 100}%;">${escapeHtml(sl.text)}</span>`;
  }
  return `<div class="diagram-canvas diagram-shapes-canvas">${body}</div>`;
}

function renderDiagramMap(data = {}) {
  const regions = Array.isArray(data.regions) ? data.regions : [];
  const pins = Array.isArray(data.pins) ? data.pins : [];
  let body = '';
  for (const r of regions) {
    if (!Array.isArray(r.points) || r.points.length < 3) continue;
    const xs = r.points.map((p) => p[0]), ys = r.points.map((p) => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const w = Math.max(1, maxX - minX), h = Math.max(1, maxY - minY);
    const clip = r.points.map((p) => `${((p[0] - minX) / w) * 100}% ${((p[1] - minY) / h) * 100}%`).join(', ');
    const cx = xs.reduce((a, b) => a + b, 0) / xs.length, cy = ys.reduce((a, b) => a + b, 0) / ys.length;
    const fillVar = DIAGRAM_ACCENT[r.fill] === DIAGRAM_ACCENT.none ? 'var(--diagram-poly-fill)' : (DIAGRAM_ACCENT[r.fill] || 'var(--diagram-poly-fill)');
    body += `<div class="diagram-shape diagram-shape-poly" style="left:${(minX / 300) * 100}%; top:${(minY / 300) * 100}%; width:${(w / 300) * 100}%; height:${(h / 300) * 100}%; clip-path: polygon(${clip}); background:${fillVar};"></div>`;
    if (r.label) body += `<span class="diagram-shape-freelabel" style="left:${(cx / 300) * 100}%; top:${(cy / 300) * 100}%;">${escapeHtml(r.label)}</span>`;
  }
  for (const p of pins) {
    body += `<div class="diagram-map-pin" style="left:${(p.x / 300) * 100}%; top:${(p.y / 300) * 100}%;">
      ${p.label ? `<span class="diagram-map-pin-label">${escapeHtml(p.label)}</span>` : ''}
    </div>`;
  }
  return `<div class="diagram-canvas diagram-map-canvas">${body}</div>`;
}

function renderDiagramTimeline(data = {}) {
  const events = Array.isArray(data.events) ? data.events : [];
  const items = events.map((e, i) => `
    <div class="diagram-timeline-event ${i % 2 === 0 ? 'above' : 'below'}" style="left:${clampNum(e.position)}%;">
      <div class="diagram-timeline-dot"></div>
      <div class="diagram-timeline-text">
        <div class="diagram-timeline-label">${escapeHtml(e.label)}</div>
        ${e.sublabel ? `<div class="diagram-timeline-sublabel">${escapeHtml(e.sublabel)}</div>` : ''}
      </div>
    </div>`).join('');
  return `<div class="diagram-timeline-wrap"><div class="diagram-timeline-line"></div>${items}</div>`;
}

function renderDiagramNumberLine(data = {}) {
  const min = clampNum(data.min, 0), max = clampNum(data.max, 10);
  const step = clampNum(data.step, 1) || 1;
  const marks = Array.isArray(data.marks) ? data.marks : [];
  const ranges = Array.isArray(data.ranges) ? data.ranges : [];
  const span = (max - min) || 1;
  const pct = (v) => ((clampNum(v) - min) / span) * 100;

  let ticks = '';
  for (let v = min; v <= max + 1e-9; v += step) {
    ticks += `<div class="diagram-numline-tick" style="left:${pct(v)}%;"><span>${Math.round(v * 100) / 100}</span></div>`;
  }
  const rangesHtml = ranges.map((r) => `<div class="diagram-numline-range diagram-numline-${r.color || 'accent1'}" style="left:${pct(r.from)}%; width:${pct(r.to) - pct(r.from)}%;"></div>`).join('');
  const marksHtml = marks.map((m) => `
    <div class="diagram-numline-mark diagram-numline-${m.color || 'accent1'}" style="left:${pct(m.value)}%;">
      ${m.label ? `<span class="diagram-numline-mark-label">${escapeHtml(m.label)}</span>` : ''}
    </div>`).join('');

  return `<div class="diagram-numline-wrap">
    <div class="diagram-numline-track">${rangesHtml}<div class="diagram-numline-baseline"></div>${ticks}${marksHtml}</div>
  </div>`;
}

function renderDiagramClock(data = {}) {
  const hour = ((clampNum(data.hour, 0) % 12) + 12) % 12;
  const minute = ((clampNum(data.minute, 0) % 60) + 60) % 60;
  const minAngle = (minute / 60) * 360;
  const hourAngle = ((hour + minute / 60) / 12) * 360;
  const ticks = Array.from({ length: 12 }, (_, i) => `
    <div class="diagram-clock-tick" style="transform: rotate(${i * 30}deg);"></div>
    <div class="diagram-clock-num" style="transform: rotate(${i * 30}deg);"><span style="transform: rotate(${-i * 30}deg);">${i === 0 ? 12 : i}</span></div>
  `).join('');
  return `<div class="diagram-clock-wrap">
    <div class="diagram-clock-face">
      ${ticks}
      <div class="diagram-clock-hand diagram-clock-hour" style="transform: rotate(${hourAngle}deg);"></div>
      <div class="diagram-clock-hand diagram-clock-minute" style="transform: rotate(${minAngle}deg);"></div>
      <div class="diagram-clock-center"></div>
    </div>
    ${data.label ? `<div class="diagram-clock-label">${escapeHtml(data.label)}</div>` : ''}
  </div>`;
}

function renderDiagramMusicStaff(data = {}) {
  const notes = Array.isArray(data.notes) ? data.notes : [];
  const clefGlyph = data.clef === 'bass' ? '𝄢' : '𝄞';
  const lines = Array.from({ length: 5 }, () => `<div class="diagram-staff-line"></div>`).join('');
  const notesHtml = notes.map((n, i) => {
    const filled = n.duration !== 'whole' && n.duration !== 'half';
    const pos = clampNum(n.position, 0);
    // position 0 = bottom line; each unit is half a line-gap, positive = up.
    const bottomPct = 50 + pos * (10);
    const leftPct = 18 + (i / Math.max(1, notes.length - 1 || 1)) * 74;
    return `<div class="diagram-staff-note" style="left:${leftPct}%; bottom:${bottomPct}%;">
      <div class="diagram-staff-notehead ${filled ? 'filled' : ''}"></div>
      ${n.duration !== 'whole' ? '<div class="diagram-staff-stem"></div>' : ''}
      ${n.label ? `<span class="diagram-staff-note-label">${escapeHtml(n.label)}</span>` : ''}
    </div>`;
  }).join('');
  return `<div class="diagram-staff-wrap">
    <span class="diagram-staff-clef">${clefGlyph}</span>
    <div class="diagram-staff-lines">${lines}${notesHtml}</div>
  </div>`;
}

function renderDiagramLabeled(data = {}) {
  const parts = Array.isArray(data.parts) ? data.parts : [];
  let body = '';
  for (const p of parts) {
    const fillVar = DIAGRAM_ACCENT[p.fill] === DIAGRAM_ACCENT.none ? 'transparent' : (DIAGRAM_ACCENT[p.fill] || 'transparent');
    const x = clampNum(p.x), y = clampNum(p.y);
    let shapeHtml = '';
    let cx = x, cy = y;
    if (p.shape === 'circle') {
      const r = clampNum(p.r, 20);
      shapeHtml = `<div class="diagram-labeled-shape diagram-labeled-circle" style="left:${((x - r) / 300) * 100}%; top:${((y - r) / 300) * 100}%; width:${(r * 2 / 300) * 100}%; height:${(r * 2 / 300) * 100}%; background:${fillVar};"></div>`;
    } else if (p.shape === 'ellipse') {
      const w = clampNum(p.w, 20), h = clampNum(p.h, 12);
      shapeHtml = `<div class="diagram-labeled-shape diagram-labeled-ellipse" style="left:${((x - w) / 300) * 100}%; top:${((y - h) / 300) * 100}%; width:${(w * 2 / 300) * 100}%; height:${(h * 2 / 300) * 100}%; background:${fillVar};"></div>`;
    } else {
      const w = clampNum(p.w, 30), h = clampNum(p.h, 20);
      shapeHtml = `<div class="diagram-labeled-shape diagram-labeled-rect" style="left:${(x / 300) * 100}%; top:${(y / 300) * 100}%; width:${(w / 300) * 100}%; height:${(h / 300) * 100}%; background:${fillVar};"></div>`;
      cx = x + w / 2; cy = y + h / 2;
    }
    body += shapeHtml;
    if (p.label && typeof p.labelX === 'number') {
      const lx = clampNum(p.labelX), ly = clampNum(p.labelY);
      const dx = lx - cx, dy = ly - cy;
      const len = Math.hypot(dx, dy);
      const angle = Math.atan2(dy, dx) * 180 / Math.PI;
      body += `<div class="diagram-labeled-leader" style="left:${(cx / 300) * 100}%; top:${(cy / 300) * 100}%; width:${(len / 300) * 100}%; transform: rotate(${angle}deg);"></div>`;
      body += `<span class="diagram-labeled-tag" style="left:${(lx / 300) * 100}%; top:${(ly / 300) * 100}%;">${escapeHtml(p.label)}</span>`;
    }
  }
  return `<div class="diagram-canvas diagram-labeled-canvas">${body}</div>`;
}

function renderDiagramGridTable(data = {}) {
  const cols = Math.max(1, Math.round(clampNum(data.cols, 4)));
  const rows = Math.max(1, Math.round(clampNum(data.rows, 4)));
  const cells = Array.isArray(data.cells) ? data.cells : [];
  const cellMap = new Map();
  for (const c of cells) {
    if (typeof c.col === 'number' && typeof c.row === 'number') cellMap.set(`${c.col},${c.row}`, c);
  }
  let body = '';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = cellMap.get(`${c},${r}`);
      const fillVar = cell ? (DIAGRAM_ACCENT[cell.fill] === DIAGRAM_ACCENT.none ? 'transparent' : (DIAGRAM_ACCENT[cell.fill] || 'transparent')) : 'transparent';
      body += `<div class="diagram-gridtable-cell" style="background:${fillVar};">
        ${cell && cell.label ? `<span class="diagram-gridtable-label">${escapeHtml(cell.label)}</span>` : ''}
        ${cell && cell.sublabel ? `<span class="diagram-gridtable-sublabel">${escapeHtml(cell.sublabel)}</span>` : ''}
      </div>`;
    }
  }
  return `<div class="diagram-gridtable-wrap" style="grid-template-columns: repeat(${cols}, 1fr); grid-template-rows: repeat(${rows}, 1fr);">${body}</div>`;
}

function renderDiagramVenn(data = {}) {
  const circles = (Array.isArray(data.circles) ? data.circles : []).slice(0, 3);
  const layoutClass = circles.length === 3 ? 'three' : 'two';
  const circlesHtml = circles.map((c, i) => `
    <div class="diagram-venn-circle diagram-venn-c${i + 1}">
      <div class="diagram-venn-label">${escapeHtml(c.label || '')}</div>
      <div class="diagram-venn-items">
        ${(Array.isArray(c.items) ? c.items : []).slice(0, 3).map((it) => `<div>${escapeHtml(it)}</div>`).join('')}
      </div>
    </div>`).join('');
  return `<div class="diagram-venn-wrap diagram-venn-${layoutClass}">${circlesHtml}</div>`;
}

function renderDiagramChart(data = {}) {
  const items = Array.isArray(data.items) ? data.items : [];
  const colors = ['var(--diagram-c1)', 'var(--diagram-c2)', 'var(--diagram-c3)', 'var(--diagram-c4)', 'var(--diagram-c5)', 'var(--diagram-c6)'];

  if (data.chartType === 'pie') {
    const total = items.reduce((s, i) => s + (clampNum(i.value, 0)), 0) || 1;
    let acc = 0;
    const stops = items.map((it, i) => {
      const start = (acc / total) * 360;
      acc += clampNum(it.value, 0);
      const end = (acc / total) * 360;
      return `${colors[i % colors.length]} ${start}deg ${end}deg`;
    }).join(', ');
    const legend = items.map((it, i) => `
      <div class="diagram-chart-legend-item">
        <span class="diagram-chart-legend-swatch" style="background:${colors[i % colors.length]};"></span>
        <span>${escapeHtml(it.label)}</span>
      </div>`).join('');
    return `<div class="diagram-chart-wrap">
      <div class="diagram-chart-pie" style="background: conic-gradient(${stops});"></div>
      <div class="diagram-chart-legend">${legend}</div>
    </div>`;
  }

  const max = Math.max(1, ...items.map((i) => clampNum(i.value, 0)));
  const bars = items.map((it, i) => `
    <div class="diagram-chart-bar-col">
      <span class="diagram-chart-bar-value">${clampNum(it.value, 0)}</span>
      <div class="diagram-chart-bar" style="height:${(clampNum(it.value, 0) / max) * 100}%; background:${colors[i % colors.length]};"></div>
      <span class="diagram-chart-bar-label">${escapeHtml(it.label)}</span>
    </div>`).join('');
  return `<div class="diagram-chart-bars">${bars}</div>`;
}

const DIAGRAM_TEMPLATE_RENDERERS = {
  chess: renderDiagramChess,
  graph: renderDiagramGraph,
  shapes: renderDiagramShapes,
  map: renderDiagramMap,
  timeline: renderDiagramTimeline,
  numberline: renderDiagramNumberLine,
  clock: renderDiagramClock,
  musicstaff: renderDiagramMusicStaff,
  labeled: renderDiagramLabeled,
  gridtable: renderDiagramGridTable,
  venn: renderDiagramVenn,
  chart: renderDiagramChart,
};

// Entry point used by renderCurrentPart(). Returns '' (falsy) if the
// template name is unknown or rendering throws, so callers can safely do
// `if (html) ...` the same way they already do for sanitizeLessonSvg().
function renderDiagramTemplate(templateName, templateData) {
  const fn = DIAGRAM_TEMPLATE_RENDERERS[templateName];
  if (!fn) return '';
  try {
    return fn(templateData || {});
  } catch (err) {
    console.error(`[renderDiagramTemplate] render failed for "${templateName}":`, err.message);
    return '';
  }
}

// ============================================================
// DIAGRAM FEEDBACK — "Report feedback" button under a rendered lesson
// diagram, opens a 4-step modal (rate -> good -> bad -> review/send) and
// emails the result via the generic Resend worker.
// ============================================================
let currentDiagramFeedbackContext = null; // set by diagramHtml() in renderLessonPart, describes the diagram currently on screen
let diagramFeedbackRating = 0;

function wireDiagramFeedbackBtn(part) {
  const btn = lessonPartContent.querySelector('[data-diagram-feedback-btn]');
  if (!btn) return;
  // Snapshot the context + part at click-time, since currentPartIndex/currentLesson
  // can change before the user actually opens the modal.
  const snapshotContext = currentDiagramFeedbackContext;
  btn.addEventListener('click', () => openDiagramFeedbackModal(part, snapshotContext));
}

function diagramFeedbackDescribeDiagram(ctx) {
  if (!ctx) return 'Unknown diagram';
  if (ctx.kind === 'image') return `Photo diagram (${ctx.imageCredit || 'Pexels'})`;
  if (ctx.kind === 'template') return `Template: ${ctx.template}`;
  return 'Raw SVG diagram';
}

// Renders the actual diagram markup so it can be shown in both the review
// step and the emailed report — not just a text label describing it.
function diagramFeedbackRenderDiagramMarkup(ctx) {
  if (!ctx) return '<i>Diagram unavailable</i>';
  if (ctx.kind === 'image') {
    const alt = escapeHtml(ctx.imageAlt || '');
    return `<img src="${ctx.imageUrl}" alt="${alt}" style="max-width:100%; border-radius:8px;" />`;
  }
  if (ctx.kind === 'template') {
    const html = renderDiagramTemplate(ctx.template, ctx.templateData);
    return html || '<i>Template failed to render</i>';
  }
  if (ctx.kind === 'svg') {
    return ctx.svgMarkup || '<i>SVG unavailable</i>';
  }
  return '<i>Diagram unavailable</i>';
}

function diagramFeedbackQuestionText(part) {
  // Question-type parts show `question`; text-type parts show `content`.
  // Report both when present instead of silently dropping one via `||`.
  const bits = [];
  if (part.question) bits.push(part.question);
  if (part.content) bits.push(part.content);
  return bits.join('\n\n');
}

function openDiagramFeedbackModal(part, ctx) {
  const overlay = document.getElementById('diagramFeedbackOverlay');
  if (!overlay) return;

  diagramFeedbackRating = 0;
  overlay.querySelectorAll('.diagram-feedback-star').forEach((s) => s.classList.remove('filled'));
  const goodInput = document.getElementById('diagramFeedbackGoodInput');
  const badInput = document.getElementById('diagramFeedbackBadInput');
  if (goodInput) goodInput.value = '';
  if (badInput) badInput.value = '';
  const errEl = document.getElementById('diagramFeedbackError');
  if (errEl) errEl.textContent = '';
  document.getElementById('diagramFeedbackStep1Next').disabled = true;

  overlay._feedbackPart = part;
  overlay._feedbackCtx = ctx;

  diagramFeedbackGoToStep(1);
  overlay.classList.add('show');
}

function closeDiagramFeedbackModal() {
  document.getElementById('diagramFeedbackOverlay')?.classList.remove('show');
}

function diagramFeedbackGoToStep(step) {
  const overlay = document.getElementById('diagramFeedbackOverlay');
  overlay.querySelectorAll('.diagram-feedback-step').forEach((el) => {
    el.classList.toggle('active', el.dataset.step === String(step));
  });
  overlay.querySelectorAll('.diagram-feedback-step-dot').forEach((dot) => {
    dot.classList.toggle('active', Number(dot.dataset.dot) <= Number(step));
  });
  if (step === 4) {
    const part = overlay._feedbackPart;
    const ctx = overlay._feedbackCtx;
    const good = document.getElementById('diagramFeedbackGoodInput').value.trim();
    const bad = document.getElementById('diagramFeedbackBadInput').value.trim();
    const recap = document.getElementById('diagramFeedbackRecap');
    recap.innerHTML = `
      <div style="margin-bottom:10px;">${diagramFeedbackRenderDiagramMarkup(ctx)}</div>
      <div><b>Rating:</b> ${diagramFeedbackRating}/5</div>
      <div><b>Question:</b> ${escapeHtml(diagramFeedbackQuestionText(part))}</div>
      <div><b>What's good:</b> ${good ? escapeHtml(good) : '<i>Not provided</i>'}</div>
      <div><b>What's bad:</b> ${bad ? escapeHtml(bad) : '<i>Not provided</i>'}</div>
    `;
  }
}

function buildDiagramFeedbackEmailHtml({ ctx, part, rating, good, bad }) {
  const lessonTitle = currentLesson?.title || currentLesson?.id || 'Unknown lesson';
  const unitLabel = Number.isInteger(currentLesson?.unitIndex) ? `Unit ${currentLesson.unitIndex + 1}` : 'Unknown unit';
  const courseTitle = activeCourse?.title || activeCourse?.id || 'Unknown course';
  const userEmail = auth.currentUser?.email || 'Unknown user';
  return `
    <h2>Diagram feedback (${rating}/5)</h2>
    <p><b>User:</b> ${escapeHtml(userEmail)}</p>
    <p><b>Course:</b> ${escapeHtml(courseTitle)} &nbsp; <b>Unit:</b> ${escapeHtml(unitLabel)} &nbsp; <b>Lesson:</b> ${escapeHtml(lessonTitle)}</p>
    <p><b>Part index:</b> ${currentPartIndex}</p>
    <p><b>Diagram (${escapeHtml(diagramFeedbackDescribeDiagram(ctx))}):</b></p>
    <div>${diagramFeedbackRenderDiagramMarkup(ctx)}</div>
    <p><b>Question:</b><br>${escapeHtml(diagramFeedbackQuestionText(part)).replace(/\n/g, '<br>')}</p>
    <p><b>What's good:</b><br>${good ? escapeHtml(good).replace(/\n/g, '<br>') : '<i>Not provided</i>'}</p>
    <p><b>What's bad:</b><br>${bad ? escapeHtml(bad).replace(/\n/g, '<br>') : '<i>Not provided</i>'}</p>
  `;
}

async function submitDiagramFeedback() {
  const overlay = document.getElementById('diagramFeedbackOverlay');
  const submitBtn = document.getElementById('diagramFeedbackSubmitBtn');
  const errEl = document.getElementById('diagramFeedbackError');
  const good = document.getElementById('diagramFeedbackGoodInput').value.trim();
  const bad = document.getElementById('diagramFeedbackBadInput').value.trim();
  const part = overlay._feedbackPart;
  const ctx = overlay._feedbackCtx;

  errEl.textContent = '';
  submitBtn.disabled = true;
  submitBtn.textContent = 'Sending…';

  try {
    const res = await fetch(EMAIL_WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        to: DIAGRAM_FEEDBACK_TO_EMAIL,
        subject: `(URGENT) Diagram feedback — ${diagramFeedbackDescribeDiagram(ctx)}`,
        html: buildDiagramFeedbackEmailHtml({ ctx, part, rating: diagramFeedbackRating, good, bad }),
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.success) throw new Error(data.error || 'Failed to send feedback');
    diagramFeedbackGoToStep('success');
  } catch (err) {
    console.error('[diagramFeedback] send failed:', err.message);
    errEl.textContent = "Couldn't send feedback — please try again.";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Send Feedback';
  }
}

(function initDiagramFeedbackModal() {
  const overlay = document.getElementById('diagramFeedbackOverlay');
  if (!overlay) return;

  overlay.querySelectorAll('.diagram-feedback-star').forEach((star) => {
    star.addEventListener('click', () => {
      diagramFeedbackRating = Number(star.dataset.value);
      overlay.querySelectorAll('.diagram-feedback-star').forEach((s) => {
        s.classList.toggle('filled', Number(s.dataset.value) <= diagramFeedbackRating);
      });
      document.getElementById('diagramFeedbackStep1Next').disabled = diagramFeedbackRating < 1;
    });
  });

  document.getElementById('diagramFeedbackStep1Next')?.addEventListener('click', () => diagramFeedbackGoToStep(2));
  document.getElementById('diagramFeedbackStep1Cancel')?.addEventListener('click', closeDiagramFeedbackModal);
  document.getElementById('diagramFeedbackStep2Next')?.addEventListener('click', () => diagramFeedbackGoToStep(3));
  document.getElementById('diagramFeedbackStep2Back')?.addEventListener('click', () => diagramFeedbackGoToStep(1));
  document.getElementById('diagramFeedbackStep3Next')?.addEventListener('click', () => diagramFeedbackGoToStep(4));
  document.getElementById('diagramFeedbackStep3Back')?.addEventListener('click', () => diagramFeedbackGoToStep(2));
  document.getElementById('diagramFeedbackStep4Back')?.addEventListener('click', () => diagramFeedbackGoToStep(3));
  document.getElementById('diagramFeedbackSubmitBtn')?.addEventListener('click', submitDiagramFeedback);
  document.getElementById('diagramFeedbackDoneBtn')?.addEventListener('click', closeDiagramFeedbackModal);
})();

// ============================================================
// STREAK
// ============================================================
async function loadStreak() {
  const u = uid();
  if (!u) return;
  const ref = doc(db, 'users', u, 'learnProfile', 'main');
  const snap = await getDoc(ref);
  if (snap.exists()) {
    learnProfile = snap.data();
    if (!Array.isArray(learnProfile.completedDates)) learnProfile.completedDates = [];
    if (typeof learnProfile.xp !== 'number') learnProfile.xp = 0;
  } else {
    learnProfile = { streak: 0, xp: 0, lastLessonDate: null, missedDaysInRow: 0, completedDates: [] };
    await setDoc(ref, learnProfile);
  }
  normalizeShopFields(learnProfile);
  normalizeReviewFields(learnProfile);
  // Weak spots / strengths now live in RTDB (reviewSpots/{uid}) instead of
  // the Firestore learnProfile doc — fetch and merge them in here.
  await loadReviewSpotsFromRtdb();
  await applyStreakDecay();
  learnStreakCount.textContent = learnProfile.streak;
  checkStreakBadges(learnProfile.streak);
  maybeShowStreakPassUsedOverlay();
}

// "Used a Streak Pass yesterday…" full-pager — fires when yesterday's date
// shows up in passDates (i.e. applyStreakDecay(), just above, auto-consumed
// a pass to cover a missed day and that missed day was yesterday, not some
// earlier gap being backfilled). Purely informational nudge; doesn't touch
// streakPassCount itself.
function maybeShowStreakPassUsedOverlay() {
  if (!Array.isArray(learnProfile.passDates) || !learnProfile.passDates.length) return;
  const y = new Date();
  y.setDate(y.getDate() - 1);
  const yesterdayStr = `${y.getFullYear()}-${String(y.getMonth() + 1).padStart(2, '0')}-${String(y.getDate()).padStart(2, '0')}`;
  if (learnProfile.passDates.includes(yesterdayStr)) {
    maybeShowOverlay('streakPassUsedYesterday');
  }
}

// Any full day with no lesson breaks the streak — UNLESS that specific
// missed date is already covered by a Streak Pass (bought from the XP
// Shop, either an individually-purchased pass consumed here on the fly,
// or a pre-applied Streak Protection date). Passes are consumed oldest
// missed day first, one per missed day, until either the days are covered
// or the streakPassCount runs out — any day left uncovered breaks the
// streak to 0.
async function applyStreakDecay() {
  if (!learnProfile.lastLessonDate) return;
  const last = new Date(learnProfile.lastLessonDate + 'T00:00:00');
  const today = new Date(todayStr() + 'T00:00:00');
  const daysSince = Math.round((today - last) / 86400000);
  const missed = Math.max(0, daysSince - 1);
  learnProfile.missedDaysInRow = missed;
  if (missed === 0) return;

  if (!Array.isArray(learnProfile.passDates)) learnProfile.passDates = [];
  const passDatesSet = new Set(learnProfile.passDates);

  // The `missed` days in question are the ones strictly between
  // lastLessonDate and today.
  const missedDateStrs = [];
  for (let i = 1; i <= missed; i++) {
    const d = new Date(last);
    d.setDate(d.getDate() + i);
    missedDateStrs.push(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`);
  }

  let uncovered = 0;
  let passesConsumed = 0;
  for (const dateStr of missedDateStrs) {
    if (passDatesSet.has(dateStr)) continue; // already covered (e.g. Streak Protection)
    if (learnProfile.streakPassCount - passesConsumed > 0) {
      passDatesSet.add(dateStr);
      passesConsumed++;
    } else {
      uncovered++;
    }
  }

  if (passesConsumed > 0 || uncovered === 0) {
    learnProfile.passDates = [...passDatesSet];
    learnProfile.streakPassCount -= passesConsumed;
    const u = uid();
    if (u) {
      await setDoc(doc(db, 'users', u, 'learnProfile', 'main'), {
        passDates: learnProfile.passDates,
        streakPassCount: learnProfile.streakPassCount,
      }, { merge: true }).catch(() => {});
    }
  }

  if (uncovered > 0) learnProfile.streak = 0;
}

function todayStr() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

async function bumpStreak() {
  const u = uid();
  if (!u) return false;
  const today = todayStr();
  if (learnProfile.lastLessonDate === today) return false; // already counted today

  learnProfile.streak = (learnProfile.streak || 0) + 1;
  learnProfile.lastLessonDate = today;
  learnProfile.missedDaysInRow = 0;
  if (!Array.isArray(learnProfile.completedDates)) learnProfile.completedDates = [];
  if (!learnProfile.completedDates.includes(today)) {
    learnProfile.completedDates.push(today);
  }

  const ref = doc(db, 'users', u, 'learnProfile', 'main');
  // Merge only the fields this function actually changed — xp is
  // deliberately excluded so this write can never clobber a concurrent
  // atomic XP increment (from a game, lesson, or badge award) with a
  // stale in-memory total.
  await setDoc(ref, {
    streak: learnProfile.streak,
    lastLessonDate: learnProfile.lastLessonDate,
    missedDaysInRow: learnProfile.missedDaysInRow,
    completedDates: learnProfile.completedDates,
  }, { merge: true });
  learnStreakCount.textContent = learnProfile.streak;
  await checkStreakBadges(learnProfile.streak);
  return true;
}

learnStreakBtn.addEventListener('click', () => {
  streakModalCount.textContent = learnProfile.streak;
  streakModalMissed.textContent = learnProfile.missedDaysInRow > 0
    ? `${learnProfile.missedDaysInRow} day${learnProfile.missedDaysInRow > 1 ? 's' : ''} missed in a row`
    : 'No missed days';
  const passCount = learnProfile.streakPassCount || 0;
  streakPassCountLabel.textContent = passCount;
  streakPassPlural.textContent = passCount === 1 ? '' : 'es';
  calendarViewDate = new Date();
  renderStreakCalendar();
  streakModalOverlay.classList.add('show');
  maybeShowOverlay('streaksWelcome');
});
streakModalCloseBtn.addEventListener('click', () => {
  streakModalOverlay.classList.remove('show');
});
streakModalShopBtn.addEventListener('click', () => {
  streakModalOverlay.classList.remove('show');
  openXpShop();
});

lessonStreakContinueBtn.addEventListener('click', () => {
  lessonStreakOverlay.classList.remove('show');
  streakModalPostLessonFlow = false;
  resumeCelebrations(advanceAfterLessonCompletion);
});

streakCalPrevBtn.addEventListener('click', () => {
  calendarViewDate = new Date(calendarViewDate.getFullYear(), calendarViewDate.getMonth() - 1, 1);
  renderStreakCalendar();
});
streakCalNextBtn.addEventListener('click', () => {
  calendarViewDate = new Date(calendarViewDate.getFullYear(), calendarViewDate.getMonth() + 1, 1);
  renderStreakCalendar();
});

function renderStreakCalendar() {
  const year = calendarViewDate.getFullYear();
  const month = calendarViewDate.getMonth();
  streakCalMonthLabel.textContent = `${MONTH_NAMES[month]} ${year}`;

  const now = new Date();
  streakCalNextBtn.disabled = (year === now.getFullYear() && month === now.getMonth());

  const firstWeekday = new Date(year, month, 1).getDay(); // 0 = Sunday
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const done = new Set(learnProfile.completedDates || []);
  const passed = new Set(learnProfile.passDates || []);
  const today = todayStr();

  let html = '';
  for (let i = 0; i < firstWeekday; i++) {
    html += `<div class="streak-cal-day empty"></div>`;
  }
  for (let d = 1; d <= daysInMonth; d++) {
    const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
    const classes = ['streak-cal-day'];
    // A Streak Pass day only shows green if it wasn't ALSO a real lesson
    // day — a day the learner actually showed up for stays orange even if
    // a pass happens to cover it (e.g. Streak Protection pre-covering the
    // next 7 days including today).
    if (passed.has(dateStr) && !done.has(dateStr)) classes.push('passed');
    else if (done.has(dateStr)) classes.push('done');
    if (dateStr === today) classes.push('today');
    html += `<div class="${classes.join(' ')}">${d}</div>`;
  }
  streakCalGrid.innerHTML = html;
}

// ============================================================
// COURSES — load + top-level render
// ============================================================
async function loadCourses() {
  const u = uid();
  if (!u) return;
  const snap = await getDocs(query(collection(db, 'users', u, 'learnCourses'), orderBy('lastOpenedAt', 'desc')));
  courses = snap.docs.map((d) => ({ id: d.id, ...d.data() }));
  activeCourse = courses[0] || null;
  checkCourseBadges(courses.length);
}

function renderCourseHomeOrEmpty(opts = {}) {
  const { resetView = true } = opts;

  learnSkeleton.style.display = 'none';

  if (!activeCourse) {
    learnEmptyState.style.display = 'flex';
    learnCourseHome.style.display = 'none';
    wrongAnswers = [];
    wrongAnswersCourseId = null;
    return;
  }
  learnEmptyState.style.display = 'none';
  learnCourseHome.style.display = 'block';
  applyCourseColor(activeCourse.color);

  // Load (once per course switch) the bank of previously-missed questions
  // this learner can review, and keep the button in sync with it.
  if (activeCourse.id !== wrongAnswersCourseId) {
    wrongAnswersCourseId = activeCourse.id;
    loadWrongAnswers(activeCourse.id);
  } else {
    updateReviewWrongAnswersBtn();
  }

  learnCourseTitle.textContent = activeCourse.title;

  const courseDone = activeCourse.currentUnitIndex >= UNITS_PER_COURSE;
  if (activeCourse.status === 'generating' && !courseDone) {
    learnCourseStatus.textContent = 'Preparing your first lesson…';
  } else if (courseDone) {
    learnCourseStatus.textContent = activeCourse.courseReviewCompleted ? 'Completed' : 'Final review';
  } else {
    learnCourseStatus.textContent = `Unit ${activeCourse.currentUnitIndex + 1} of ${UNITS_PER_COURSE}`;
  }

  if (resetView) viewedUnitIndex = null;
  renderCourseBody();
}

function applyCourseColor(hex) {
  learnCourseHome.style.setProperty('--cc', hex);
  learnCourseHome.style.setProperty('--cc-pale', hexToPale(hex));
}
function hexToPale(hex) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, 0.16)`;
}

// Decides what to show in the course body: the all-units overview, a single
// unit's lesson path, or the final course-review card (once all 10 units
// are done).
function renderCourseBody() {
  const c = activeCourse;
  const courseDone = c.currentUnitIndex >= UNITS_PER_COURSE;

  learnCourseReviewCard.style.display = courseDone ? 'flex' : 'none';

  if (courseDone) {
    learnUnitsList.style.display = 'none';
    learnCourseDescription.style.display = 'none';
    learnUnitPathView.style.display = 'none';

    if (c.courseReviewCompleted) {
      learnCourseReviewTitle.textContent = 'Course Complete! 🎉';
      learnCourseReviewSub.textContent = `You finished "${c.title}". Nice work!`;
      learnCourseReviewBtn.style.display = 'none';
    } else {
      learnCourseReviewTitle.textContent = 'Course Review';
      learnCourseReviewSub.textContent = '15 questions covering everything from the whole course.';
      learnCourseReviewBtn.textContent = 'Start Course Review';
      learnCourseReviewBtn.style.display = 'inline-flex';
    }
    return;
  }

  if (viewedUnitIndex !== null && viewedUnitIndex <= c.currentUnitIndex) {
    learnUnitsList.style.display = 'none';
    learnCourseDescription.style.display = 'none';
    learnUnitPathView.style.display = 'block';
    const unit = c.units[viewedUnitIndex];
    unitPathTitle.textContent = `Unit ${viewedUnitIndex + 1}: ${unit.title}`;
    unitPathDescription.textContent = unit.description || '';
    renderLessonPath(viewedUnitIndex);
  } else {
    learnUnitPathView.style.display = 'none';
    learnUnitsList.style.display = 'grid';
    learnCourseDescription.textContent = c.description || '';
    learnCourseDescription.style.display = c.description ? 'block' : 'none';
    renderUnitsList();
  }
}

// All-units overview — lets the learner pick and choose between any unit
// they've already unlocked. Units beyond their current progress stay locked.
function renderUnitsList() {
  const c = activeCourse;
  learnUnitsList.innerHTML = '';
  for (let i = 0; i < UNITS_PER_COURSE; i++) {
    const unit = c.units[i];
    if (!unit) continue;
    const state = i < c.currentUnitIndex ? 'completed' : (i === c.currentUnitIndex ? 'current' : 'locked');
    const card = document.createElement('button');
    card.className = `unit-card ${state}`;
    card.disabled = state === 'locked';
    const iconName = state === 'completed' ? 'check_circle' : (state === 'locked' ? 'lock' : 'play_circle');
    card.innerHTML = `
      <div class="unit-card-icon"><span class="material-symbols-outlined">${iconName}</span></div>
      <div class="unit-card-body">
        <div class="unit-card-num">Unit ${i + 1}</div>
        <div class="unit-card-title">${escapeHtml(unit.title)}</div>
      </div>
    `;
    if (state !== 'locked') {
      card.addEventListener('click', () => openUnitPath(i));
    }
    learnUnitsList.appendChild(card);
  }
}

function openUnitPath(unitIndex) {
  viewedUnitIndex = unitIndex;
  renderCourseBody();
}

unitPathBackBtn.addEventListener('click', () => {
  viewedUnitIndex = null;
  renderCourseBody();
});

// Renders the lesson-node path for a single unit. Units before the learner's
// current unit are fully completed (and replayable); the current unit is
// gated lesson-by-lesson; later units are never reachable from here.
function renderLessonPath(unitIndex) {
  const c = activeCourse;
  const isPastUnit = unitIndex < c.currentUnitIndex;
  const frontier = isPastUnit ? LESSONS_PER_UNIT : c.currentLessonIndex;

  learnLessonPath.innerHTML = '';
  for (let i = 0; i < LESSONS_PER_UNIT; i++) {
    const isReview = i === LESSONS_PER_UNIT - 1;
    const btn = document.createElement('button');
    btn.className = 'lesson-node' + (isReview ? ' review' : '');

    if (i < frontier) btn.classList.add('completed');
    else if (i === frontier) btn.classList.add('available');
    else btn.classList.add('locked');

    const iconName = isReview ? 'emoji_events' : (i < frontier ? 'check' : 'star');
    btn.innerHTML = `<span class="material-symbols-outlined">${iconName}</span>`;
    btn.style.transform = `translateX(${lessonPathOffset(i)}px)`;
    btn.addEventListener('click', () => {
      if (i > frontier) return; // future lesson, still locked
      openLessonStartModal(unitIndex, i);
    });
    learnLessonPath.appendChild(btn);
  }
}

// Gentle side-to-side wave so the path zig-zags like a trail instead of
// stacking straight down the middle.
const LESSON_PATH_WAVE = [0, 55, 80, 55, 0, -55, -80, -55];
function lessonPathOffset(i) {
  return LESSON_PATH_WAVE[i % LESSON_PATH_WAVE.length];
}

// ============================================================
// WRONG-ANSWER REVIEW (missed questions stored in Firestore, per course)
// ============================================================

// Loads every question this learner has ever missed in `courseId` (that
// hasn't since been cleared by a correct re-answer) and refreshes the button.
async function loadWrongAnswers(courseId) {
  try {
    const u = uid();
    if (!u) return;
    const snap = await getDocs(collection(db, 'users', u, 'learnCourses', courseId, 'wrongAnswers'));
    // Bail out quietly if the active course changed while this was in flight.
    if (activeCourse?.id !== courseId) return;
    wrongAnswers = snap.docs.map((d) => ({ id: d.id, ...d.data() }));
    updateReviewWrongAnswersBtn();
  } catch (err) {
    console.error('Failed to load wrong answers:', err);
  }
}

// Records a missed question to Firestore so it can resurface later. Skips
// silently while already inside a wrong-answer review (it's in the bank already).
async function saveWrongAnswer(part) {
  try {
    const u = uid();
    if (!u || !activeCourse) return;
    const wrongAnswersCol = collection(db, 'users', u, 'learnCourses', activeCourse.id, 'wrongAnswers');
    const wrongRef = doc(wrongAnswersCol);
    const entry = {
      question: part.question,
      choices: part.choices,
      correctIndex: part.correctIndex,
      lessonTitle: currentLesson?.lessonTitle || null,
      unitIndex: currentLesson?.unitIndex ?? null,
      // Diagram fields — a part carries at most one of these (see
      // renderCurrentPart's diagramHtml()). Saved as-is (undefined fields
      // just don't get written) so a reconstructed review question shows
      // the same photo/diagram the learner originally saw it with, instead
      // of silently losing it.
      imageUrl: part.imageUrl || null,
      imageAlt: part.imageAlt || null,
      imageCredit: part.imageCredit || null,
      template: part.template || null,
      templateData: part.templateData || null,
      svg: part.svg || null,
      createdAt: serverTimestamp(),
    };
    await setDoc(wrongRef, entry);
    if (activeCourse.id === wrongAnswersCourseId) {
      wrongAnswers.push({ id: wrongRef.id, ...entry });
      updateReviewWrongAnswersBtn();
    }
  } catch (err) {
    console.error('Failed to save wrong answer:', err);
  }
}

// Clears a question from the bank once the learner answers it correctly
// during a review session.
async function removeWrongAnswer(docId) {
  try {
    const u = uid();
    if (!u || !activeCourse) return;
    await deleteDoc(doc(db, 'users', u, 'learnCourses', activeCourse.id, 'wrongAnswers', docId));
    wrongAnswers = wrongAnswers.filter((w) => w.id !== docId);
    updateReviewWrongAnswersBtn();
  } catch (err) {
    console.error('Failed to remove wrong answer:', err);
  }
}

function updateReviewWrongAnswersBtn() {
  if (!reviewWrongAnswersBtn) return;
  // Repointed to the new Review Page — hidden entirely when Review
  // Lessons are turned off in Additional Settings (see
  // getReviewLessonsEnabled()).
  reviewWrongAnswersBtn.style.display = getReviewLessonsEnabled() ? 'inline-flex' : 'none';
  reviewWrongAnswersLabel.textContent = 'Review Page';
}

const noMistakesModalOverlay = document.getElementById('noMistakesModalOverlay');
const noMistakesModalBody = document.getElementById('noMistakesModalBody');
const noMistakesModalCloseBtn = document.getElementById('noMistakesModalCloseBtn');
noMistakesModalCloseBtn?.addEventListener('click', () => noMistakesModalOverlay.classList.remove('show'));

reviewWrongAnswersBtn.addEventListener('click', openReviewPage);

reviewPageDailyLessonBtn?.addEventListener('click', async () => {
  if (getDailyLessonDoneToday()) return;
  await openComboLesson();
});

// ============================================================
// REVIEW PAGE — AI-picked weak spots & strengths (learnProfile.weakSpots /
// .strengths, backed by RTDB at reviewSpots/{uid} — NOT the Firestore
// learnProfile doc). Free: per-topic review buttons + Strengths full
// review (capped at 3 free button-reviews/day, local-midnight reset).
// Premium (or an XP Shop "Full Personalized Review" credit): Full Weak
// Spot Review and Full Personalized Review, unlimited for Premium.
// ============================================================
function normalizeReviewFields(learnProfile) {
  if (!Array.isArray(learnProfile.weakSpots)) learnProfile.weakSpots = [];
  if (!Array.isArray(learnProfile.strengths)) learnProfile.strengths = [];
  return learnProfile;
}

// Fetches weakSpots/strengths from RTDB (reviewSpots/{uid}) and merges them
// onto the in-memory learnProfile object. Called once during loadStreak();
// after that, learnProfile.weakSpots/.strengths stay in sync locally and
// every write goes back out to the same RTDB path (see deleteReviewSpot
// and maybeGenerateReviewSpots below).
async function loadReviewSpotsFromRtdb() {
  const u = uid();
  if (!u) return;
  try {
    const snap = await rtdbGet(rtdbRef(rtdb, `reviewSpots/${u}`));
    const data = snap.exists() ? snap.val() : null;
    learnProfile.weakSpots = Array.isArray(data?.weakSpots) ? data.weakSpots : [];
    learnProfile.strengths = Array.isArray(data?.strengths) ? data.strengths : [];
    learnProfile.reviewAiSummary = typeof data?.aiSummary === 'string' ? data.aiSummary : '';
  } catch (err) {
    console.error('Failed to load weak/strong spots from RTDB:', err);
    learnProfile.weakSpots = learnProfile.weakSpots || [];
    learnProfile.strengths = learnProfile.strengths || [];
    learnProfile.reviewAiSummary = learnProfile.reviewAiSummary || '';
  }
}

// Asks the text worker for a short paragraph analyzing patterns across
// every current weak spot + strength together (not per-spot) — e.g. a
// subject that keeps showing up as a weakness, or a skill that's
// consistently strong. Regenerated (not appended to) every time the
// weak/strong spot list changes, and persisted to RTDB at
// reviewSpots/{uid}/aiSummary so it survives reloads like the spots do.
async function regenerateReviewAiSummary() {
  const u = uid();
  if (!u) return;

  const weakSpots = learnProfile.weakSpots || [];
  const strengths = learnProfile.strengths || [];
  if (!weakSpots.length && !strengths.length) {
    learnProfile.reviewAiSummary = '';
    renderReviewAiSummary();
    await rtdbSet(rtdbRef(rtdb, `reviewSpots/${u}/aiSummary`), '').catch(() => {});
    return;
  }

  const listFor = (items) => items.map((s, i) => `${i + 1}. "${s.title}" — ${s.description || 'no description'}`).join('\n') || '(none yet)';
  const openerRule = strengths.length
    ? `The first sentence MUST start with the exact words "You are doing great with" followed by their strongest pattern/theme.`
    : `There are no strengths yet, so the first sentence MUST start with the exact words "You are doing great with" followed by the most promising early pattern you can find in the weak spots' descriptions (something they're closer to getting, or attempting well) — do not invent an unrelated strength.`;
  const prompt = `You analyze a learner's overall performance patterns for a direct, second-person summary on their own review page. This app is used by learners of any age — do not assume they are a child, and do not refer to "your child" or address a parent.
Weak spots:
${listFor(weakSpots)}

Strengths:
${listFor(strengths)}

Write EXACTLY 2 sentences (plain text, no markdown, no headers) analyzing the OVERALL pattern across all of these together — e.g. a subject or skill that keeps showing up as a weakness, or a theme connecting several strengths. Speak directly to the learner as "you". ${openerRule} The second sentence should name the pattern in their weak spots to focus on next. Do not list the individual spots back verbatim — synthesize a pattern.
Respond with ONLY those 2 sentences, nothing else.`;

  try {
    const res = await fetch(`${TEXT_WORKER_URL}?prompt=${encodeURIComponent(prompt)}`);
    const data = await res.json();
    const summary = (data?.response || '').trim();
    if (!summary) return;
    learnProfile.reviewAiSummary = summary;
    renderReviewAiSummary();
    await rtdbSet(rtdbRef(rtdb, `reviewSpots/${u}/aiSummary`), summary).catch((err) => {
      console.error('Failed to save review AI summary:', err);
    });
  } catch (err) {
    console.error('Failed to generate review AI summary:', err);
  }
}

function renderReviewAiSummary() {
  const summary = learnProfile.reviewAiSummary || '';
  if (!summary) {
    reviewAiSummary.style.display = 'none';
    return;
  }
  reviewAiSummaryText.textContent = summary;
  reviewAiSummary.style.display = 'flex';
}

export function openReviewPage() {
  if (!getReviewLessonsEnabled()) {
    return;
  }
  document.querySelector('.nav-btn[data-page="review"]')?.click();
}
function renderReviewPage() {
  const weakSpots = learnProfile.weakSpots || [];
  const strengths = learnProfile.strengths || [];
  const premiumAccess = isPremium() || hasBankedPremiumPerks(learnProfile);
  const hasCredit = (learnProfile.purchasedReviewCredits || 0) > 0;

  renderReviewAiSummary();

  reviewPageFullWeakBtn.disabled = !weakSpots.length;
  reviewPageFullAllBtn.disabled = !weakSpots.length && !strengths.length;

  reviewWeakSpotsList.innerHTML = weakSpots.length
    ? weakSpots.map((s) => reviewSpotCardHtml(s, 'weak')).join('')
    : `<div class="review-spot-empty">No weak spots yet — keep taking lessons!</div>`;
  reviewStrengthsList.innerHTML = strengths.length
    ? strengths.map((s) => reviewSpotCardHtml(s, 'strong')).join('')
    : `<div class="review-spot-empty">No strengths yet — keep taking lessons!</div>`;
  reviewWeakSpotsList.querySelectorAll('.review-spot-card').forEach((card) => window.KLLAnim?.popIn(card));
  reviewStrengthsList.querySelectorAll('.review-spot-card').forEach((card) => window.KLLAnim?.popIn(card));

  reviewWeakSpotsList.querySelectorAll('[data-spot-review-btn]').forEach((btn) => {
    btn.addEventListener('click', () => startTopicReview(btn.dataset.spotId, 'weak'));
  });
  reviewStrengthsList.querySelectorAll('[data-spot-review-btn]').forEach((btn) => {
    btn.addEventListener('click', () => startTopicReview(btn.dataset.spotId, 'strong'));
  });

  reviewWeakSpotsList.querySelectorAll('[data-spot-review-btn]').forEach((btn) => {
    btn.addEventListener('click', () => startTopicReview(btn.dataset.spotId, 'weak'));
  });
  reviewStrengthsList.querySelectorAll('[data-spot-review-btn]').forEach((btn) => {
    btn.addEventListener('click', () => startTopicReview(btn.dataset.spotId, 'strong'));
  });
  reviewWeakSpotsList.querySelectorAll('[data-spot-delete-btn]').forEach((btn) => {
    btn.addEventListener('click', () => deleteReviewSpot(btn.dataset.spotId, 'weak'));
  });
  reviewStrengthsList.querySelectorAll('[data-spot-delete-btn]').forEach((btn) => {
    btn.addEventListener('click', () => deleteReviewSpot(btn.dataset.spotId, 'strong'));
  });

  // Daily Lesson entry, moved onto the Review Page per spec — greyed out
  // once done today (saved locally only, per spec).
  const doneToday = getDailyLessonDoneToday();
  reviewPageDailyLessonBtn.classList.toggle('done', doneToday);
  reviewPageDailyLessonBtn.disabled = doneToday;
  reviewPageDailyLessonTitle.textContent = doneToday ? 'Daily Lesson — done for today!' : 'Daily Lesson';

  if (premiumAccess) {
    reviewPageUsesNote.textContent = 'Unlimited reviews — thanks to Premium!';
  } else {
    const left = Math.max(0, REVIEW_FREE_USES_PER_DAY - getReviewFreeUsesToday());
    reviewPageUsesNote.textContent = hasCredit
      ? `${left} free topic review${left === 1 ? '' : 's'} left today · you also have ${learnProfile.purchasedReviewCredits} Full Review credit${learnProfile.purchasedReviewCredits === 1 ? '' : 's'}`
      : `${left} free topic review${left === 1 ? '' : 's'} left today`;
  }
}
// main.js's switchPage() calls window.renderReviewPage?.() when the Review
// nav tab becomes active (same pattern as refreshHome() for Home) — expose
// it here since it's otherwise a local, unexported function.
window.renderReviewPage = renderReviewPage;

function reviewSpotCardHtml(spot, kind) {
  return `
    <div class="review-spot-card ${kind}">
      <div class="review-spot-body">
        <div class="review-spot-title">${escapeHtml(spot.title)}</div>
        <div class="review-spot-desc">${escapeHtml(spot.description || '')}</div>
      </div>
      <div class="review-spot-actions">
        <button type="button" class="review-spot-btn" data-spot-review-btn data-spot-id="${spot.id}">Review</button>
        <button type="button" class="review-spot-delete-btn" data-spot-delete-btn data-spot-id="${spot.id}" aria-label="Delete">
          <span class="material-symbols-outlined">delete</span>
        </button>
      </div>
    </div>
  `;
}

// Small per-topic review (+15 XP) — free, but capped at 3/day total across
// ALL small reviews (weak or strength) for free accounts; unlimited for
// Premium/banked-premium-day. Generates 5 fresh questions from the spot's
// title/description/example question via the Learn Worker.
async function startTopicReview(spotId, kind) {
  const list = kind === 'weak' ? (learnProfile.weakSpots || []) : (learnProfile.strengths || []);
  const spot = list.find((s) => s.id === spotId);
  if (!spot) return;

  const premiumAccess = isPremium() || hasBankedPremiumPerks(learnProfile);
  if (!premiumAccess && !hasFreeReviewUseLeft()) {
    openPaywall({ reason: "You've used all 3 free reviews today. Upgrade to Premium for unlimited reviews." });
    return;
  }

  const btn = document.querySelector(`[data-spot-review-btn][data-spot-id="${spotId}"]`);
  if (btn) { btn.disabled = true; btn.textContent = 'Loading…'; }
  try {
    const parts = await generateReviewQuestions([{ ...spot, _kind: kind }], 5);
    if (!parts.length) throw new Error('Could not generate review questions.');
    if (!premiumAccess) bumpReviewFreeUsesToday();
    startReviewLesson(parts, `Review: ${spot.title}`, 15, { progressColorMap: buildProgressColorMap(parts) });
  } catch (err) {
    console.error('Topic review generation failed:', err);
    if (btn) { btn.disabled = false; btn.textContent = 'Review'; }
  }
}

async function deleteReviewSpot(spotId, kind) {
  const key = kind === 'weak' ? 'weakSpots' : 'strengths';
  learnProfile[key] = (learnProfile[key] || []).filter((s) => s.id !== spotId);
  renderReviewPage();

  const u = uid();
  if (!u) return;
  await rtdbSet(rtdbRef(rtdb, `reviewSpots/${u}/${key}`), learnProfile[key]).catch((err) => {
    console.error('Failed to delete review spot:', err);
  });
  regenerateReviewAiSummary();
}

// Full Weak Spot Review — Premium (or a spent XP Shop credit), +20 XP.
reviewPageFullWeakBtn?.addEventListener('click', async () => {
  const weakSpots = learnProfile.weakSpots || [];
  if (!weakSpots.length) return;
  if (!(await gateFullReview())) return;
  reviewPageFullWeakBtn.disabled = true;
  const prevText = reviewPageFullWeakBtn.querySelector('.review-page-full-title').textContent;
  try {
    const parts = await generateReviewQuestions(weakSpots.map((s) => ({ ...s, _kind: 'weak' })), 10);
    if (!parts.length) throw new Error('Could not generate review questions.');
    startReviewLesson(parts, 'Full Weak Spot Review', 20, { progressColorMap: buildProgressColorMap(parts) });
  } catch (err) {
    console.error('Full Weak Spot Review generation failed:', err);
  } finally {
    reviewPageFullWeakBtn.disabled = !(learnProfile.weakSpots || []).length;
  }
});

// Full Personalized Review — weaknesses AND strengths mixed, Premium (or a
// spent XP Shop credit), +25 XP.
reviewPageFullAllBtn?.addEventListener('click', async () => {
  const weakSpots = learnProfile.weakSpots || [];
  const strengths = learnProfile.strengths || [];
  if (!weakSpots.length && !strengths.length) return;
  if (!(await gateFullReview())) return;
  reviewPageFullAllBtn.disabled = true;
  try {
    const parts = await generateReviewQuestions(
      [...weakSpots.map((s) => ({ ...s, _kind: 'weak' })), ...strengths.map((s) => ({ ...s, _kind: 'strong' }))],
      10
    );
    if (!parts.length) throw new Error('Could not generate review questions.');
    startReviewLesson(parts, 'Full Personalized Review', 25, { progressColorMap: buildProgressColorMap(parts) });
  } catch (err) {
    console.error('Full Personalized Review generation failed:', err);
  } finally {
    reviewPageFullAllBtn.disabled = !(learnProfile.weakSpots || []).length && !(learnProfile.strengths || []).length;
  }
});

// Premium/banked-premium-day gets unlimited Full Reviews; otherwise a
// purchased XP Shop credit is required and consumed on use. Returns true
// if the review is allowed to proceed.
async function gateFullReview() {
  const premiumAccess = isPremium() || hasBankedPremiumPerks(learnProfile);
  if (premiumAccess) return true;
  const hasCredit = (learnProfile.purchasedReviewCredits || 0) > 0;
  if (!hasCredit) {
    openPaywall({ reason: 'Full Reviews are a Premium feature. You can also buy one in the XP Shop.' });
    return false;
  }
  _consumingPurchasedReview = true;
  return true;
}

// Tags each generated part with which spot (weak/strong) it came from, so
// the lesson-view progress bar can render orange (weak) / green (strong)
// segments per-question — see renderCurrentPart()'s progress fill.
function buildProgressColorMap(parts) {
  return parts.map((p) => p._reviewKind === 'weak' ? '#FF8A2B' : '#2FA84F');
}

// The text worker (textonlygroqfast) is a plain "prompt in, prose out"
// endpoint with no structured-output support — every JSON-shaped call
// below has to explicitly demand JSON-only in the prompt, then defensively
// strip any ```json fences / stray prose the model still wraps around it
// before parsing. Throws if the response genuinely isn't valid JSON.
async function fetchWorkerJson(prompt) {
  const res = await fetch(`${TEXT_WORKER_URL}?prompt=${encodeURIComponent(prompt)}`);
  const data = await res.json();
  const raw = (data?.response || '').trim();
  const cleaned = raw.replace(/^```json\s*/i, '').replace(/^```\s*/, '').replace(/```\s*$/, '').trim();
  return JSON.parse(cleaned);
}

// Generates `count` fresh multiple-choice questions from a list of
// weak/strong spots (title + description + exampleQuestion each), tagged
// with which spot kind each question came from — used by every review
// flow on this page instead of replaying stored questions verbatim.
async function generateReviewQuestions(spots, count) {
  const spotList = spots.map((s, i) => `${i + 1}. [${s._kind || 'weak'}] "${s.title}" — ${s.description || 'no description'}${s.exampleQuestion ? ` (example: ${s.exampleQuestion})` : ''}`).join('\n');
  const prompt = `You write multiple-choice quiz questions for a kids' learning app.
Given these topics the learner needs to review:
${spotList}

Write exactly ${count} multiple-choice questions total, drawn from these topics (spread across all of them, roughly proportional to how many topics are listed). Each question needs exactly 4 answer choices with exactly one correct answer.

Respond with ONLY raw JSON (no markdown fences, no commentary), in exactly this shape:
{"parts":[{"question":"...","choices":["...","...","...","..."],"correctIndex":0,"kind":"weak"}]}

"kind" must be either "weak" or "strong" — whichever topic list that specific question came from (weak spots are "weak", strengths are "strong"). "correctIndex" is the 0-based index into "choices" of the correct answer.`;

  const data = await fetchWorkerJson(prompt);
  const rawParts = Array.isArray(data.parts) ? data.parts : [];
  return rawParts
    .filter((p) => p && p.question && Array.isArray(p.choices) && p.choices.length >= 2 && typeof p.correctIndex === 'number')
    .map((p) => ({
      type: 'question',
      question: p.question,
      choices: p.choices,
      correctIndex: p.correctIndex,
      _reviewKind: p.kind === 'strong' ? 'strong' : 'weak',
    }));
}

// Called at the end of every regular (non-review, non-Daily-Lesson) lesson
// with that lesson's question/answer log. Asks the text worker to pick
// exactly 1 weak spot and 1 strength (each a 1-4 word title, short
// description, and an example question the learner actually got
// wrong/right), then prepends them to learnProfile.weakSpots/.strengths,
// capped at the latest 5 each (oldest dropped).
async function maybeGenerateReviewSpots(answerLog) {
  const u = uid();
  if (!u || !activeCourse) return;

  const answerSummary = answerLog.map((a, i) =>
    `${i + 1}. Q: "${a.question}" — learner answered "${a.choices[a.selectedIndex]}" (${a.isCorrect ? 'CORRECT' : 'WRONG'}, correct answer was "${a.choices[a.correctIndex]}")`
  ).join('\n');

  const prompt = `You analyze a kid's quiz performance in a course called "${activeCourse.title}" and pick one weak spot and one strength.
Here's every question from their lesson:
${answerSummary}

Pick exactly 1 topic they struggled with (a weak spot, based on questions they got wrong) and exactly 1 topic they're doing well on (a strength, based on questions they got right). If there are no wrong answers, still infer a weak spot as a topic that could use more practice; if there are no right answers, still infer a strength as a topic showing the most promise.

For each, give a 1-4 word title, a short one-sentence description that naturally mentions the course name ("${activeCourse.title}"), and copy the exact example question text (one of the questions above) that best represents it.
Respond with ONLY raw JSON (no markdown fences, no commentary), in exactly this shape:
{"weakSpot":{"title":"...","description":"...","exampleQuestion":"..."},"strength":{"title":"...","description":"...","exampleQuestion":"..."}}`;

  const data = await fetchWorkerJson(prompt);

  const now = Date.now();
  const updates = {};

  if (data.weakSpot && data.weakSpot.title) {
    const entry = {
      id: `ws_${now}_${Math.random().toString(36).slice(2, 8)}`,
      title: data.weakSpot.title,
      description: data.weakSpot.description || '',
      exampleQuestion: data.weakSpot.exampleQuestion || '',
      courseId: activeCourse.id,
      createdAt: now,
    };
    learnProfile.weakSpots = [entry, ...(learnProfile.weakSpots || [])].slice(0, 5);
    updates.weakSpots = learnProfile.weakSpots;
  }
  if (data.strength && data.strength.title) {
    const entry = {
      id: `st_${now}_${Math.random().toString(36).slice(2, 8)}`,
      title: data.strength.title,
      description: data.strength.description || '',
      exampleQuestion: data.strength.exampleQuestion || '',
      courseId: activeCourse.id,
      createdAt: now,
    };
    learnProfile.strengths = [entry, ...(learnProfile.strengths || [])].slice(0, 5);
    updates.strengths = learnProfile.strengths;
  }

  if (Object.keys(updates).length) {
    await Promise.all(
      Object.entries(updates).map(([key, value]) =>
        rtdbSet(rtdbRef(rtdb, `reviewSpots/${u}/${key}`), value)
      )
    ).catch((err) => {
      console.error('Failed to save weak/strong spots:', err);
    });
    regenerateReviewAiSummary();
  }
}

// Builds and plays a synthetic review lesson — same "not backed by a real
// lesson doc" pattern as the old Personalized Review / Combo Lesson.
// `xpPerCorrect` is a flat amount, NOT the normal streak-scaled gain (see
// checkAnswer()'s isReviewLesson branch).
function startReviewLesson(parts, title, xpPerCorrect, opts = {}) {
  currentLesson = {
    id: 'reviewLesson',
    isWrongAnswerReview: true, // reuses "no course-doc write" finishLesson() path
    isReviewLesson: true,
    reviewXpPerCorrect: xpPerCorrect,
    reviewProgressColorMap: opts.progressColorMap || null,
    isCourseReview: false,
    isUnitReview: false,
    lessonTitle: title,
    parts,
  };
  startLessonView();
}

// ============================================================
// COMBO LESSON — Premium only. Mixes questions from every course the
// learner has, weighted toward their biggest mistakes (most-missed
// questions first), into one lesson. Distinct from Personalized Review,
// which only ever pulls from the single active course's wrongAnswers.
//
// Built the same "synthetic lesson, not backed by a real lesson doc" way
// startWrongAnswersReview() is — see finishLesson()'s isWrongAnswerReview
// branch, which this reuses wholesale (no XP/streak, no course-doc write).
// The one thing that branch does that doesn't generalize as-is is
// removeWrongAnswer(docId), which hardcodes activeCourse.id — Combo Lesson
// mixes questions from several courses, so each part needs to carry which
// course it actually came from. See _wrongAnswerCourseId below and
// removeWrongAnswerFromCourse().
// ============================================================

// Pulls every course's wrongAnswers subcollection in parallel and returns
// them flattened, each tagged with which course + how many total mistakes
// that course has (so "biggest mistakes" can be weighted toward courses
// with the most to review, not just shuffled uniformly across all of them).
async function collectAllCoursesWrongAnswers() {
  const u = uid();
  if (!u || !courses.length) return [];
  const perCourse = await Promise.all(courses.map(async (c) => {
    try {
      const snap = await getDocs(collection(db, 'users', u, 'learnCourses', c.id, 'wrongAnswers'));
      return snap.docs.map((d) => ({ id: d.id, courseId: c.id, courseTitle: c.title, ...d.data() }));
    } catch (err) {
      console.error(`Failed to load wrong answers for course ${c.id}:`, err);
      return [];
    }
  }));
  return perCourse.flat();
}

// Separate, already-deployed worker — plain GET, ?prompt=..., returns
// { response: "..." }. No changes needed to it; this is the only place in
// the app that calls it so far.
const TEXT_WORKER_URL = 'https://textonlygroqfast.nameless-cherry-998c.workers.dev/';

// ============================================================
// WEEKLY REVIEW DATA — feeds the Premium-only "first open of the week"
// recap overlay (see main.js's maybeShowWeeklyReviewOverlay). Pulls the
// raw numbers from stuff already loaded client-side (no new Firestore
// reads beyond what Combo Lesson already does), then asks the
// textonlygroqfast worker to turn them into one encouraging sentence — the
// numbers themselves are still shown as plain stats in the overlay too, so
// a failed/slow AI call never blocks those from displaying.
// ============================================================
export async function getWeeklyReviewData(daysActive) {
  const allWrong = await collectAllCoursesWrongAnswers();
  // Group mistakes by course so the overlay can call out the toughest
  // course by name, not just a raw total.
  const byCourse = {};
  allWrong.forEach((w) => {
    byCourse[w.courseId] = (byCourse[w.courseId] || 0) + 1;
  });
  let toughestCourse = null;
  Object.entries(byCourse).forEach(([courseId, count]) => {
    if (!toughestCourse || count > toughestCourse.count) {
      const c = courses.find((c) => c.id === courseId);
      toughestCourse = { title: c?.title || 'a course', count };
    }
  });
  const stats = {
    xp: learnProfile?.xp || 0,
    streak: learnProfile?.streak ?? 0,
    courseCount: courses.length,
    totalMistakes: allWrong.length,
    toughestCourse, // { title, count } or null if no mistakes anywhere
  };
  return { ...stats, recap: await generateWeeklyRecapText(stats, daysActive) };
}

// Kept intentionally short (1-2 sentences) and parent-facing — this reads
// on the overlay, not inside a lesson, so no strict word-count/format
// constraints from the Learn Worker's usual prompt conventions apply here.
// daysActive (optional): passed when this is a brand-new account's very
// first recap, generated before they've actually had a full week — the
// prompt then talks about "their first N days" instead of "this week" so
// the wording doesn't imply more history than actually exists yet.
async function generateWeeklyRecapText(stats, daysActive) {
  const courseNames = courses.map((c) => c.title).join(', ') || 'no courses yet';
  const periodPhrase = daysActive ? `your first ${daysActive} day${daysActive === 1 ? '' : 's'} on the app` : 'this week';
const prompt = `You write short, warm progress recaps for a learning app. `
  + `Write exactly 1-2 sentences (no more), friendly and encouraging, directly addressing the learner as "you" — never "they" or "their". Summarize ${periodPhrase}. `
  + `Do not use markdown, headers, or bullet points — plain sentences only. `
  + `Stats: total XP so far is ${stats.xp}, current streak is ${stats.streak} day(s), `
  + `you have ${stats.courseCount} course(s) (${courseNames}), `
  + `and you currently have ${stats.totalMistakes} question(s) saved to review`
  + (stats.toughestCourse ? `, mostly from "${stats.toughestCourse.title}" (${stats.toughestCourse.count} of them)` : '')
  + `. If totalMistakes is 0, praise the learner for being all caught up instead of mentioning mistakes.`;
  try {
    const res = await fetch(`${TEXT_WORKER_URL}?prompt=${encodeURIComponent(prompt)}`);
    const data = await res.json();
    const text = data?.response?.trim();
    return text || null; // fall through to the overlay's own templated fallback
  } catch (err) {
    console.warn('Weekly recap generation failed, overlay will fall back to a templated line:', err);
    return null;
  }
}

function startComboLesson(allWrong) {
  // Biggest mistakes first: courses with more missed questions contribute
  // more of the mix, rather than every course getting an equal slice
  // regardless of how many mistakes it actually has.
  const picked = shuffleArray([...allWrong]).slice(0, 10);
  const parts = picked.map((w) => ({
    type: 'question',
    question: w.question,
    choices: w.choices,
    correctIndex: w.correctIndex,
    imageUrl: w.imageUrl || null,
    imageAlt: w.imageAlt || null,
    imageCredit: w.imageCredit || null,
    template: w.template || null,
    templateData: w.templateData || null,
    svg: w.svg || null,
    _wrongAnswerDocId: w.id,
    _wrongAnswerCourseId: w.courseId,
  }));
  currentLesson = {
    id: 'dailyLesson',
    // Daily Lesson (renamed from Combo Lesson) is still not backed by a
    // real lesson doc, but per spec it DOES earn flat XP and CAN advance
    // the streak — so, unlike the old Combo Lesson, it does NOT reuse the
    // isWrongAnswerReview "no XP/streak" finishLesson() path.
    isWrongAnswerReview: false,
    isDailyLesson: true,
    isComboLesson: true,
    isCourseReview: false,
    isUnitReview: false,
    lessonTitle: 'Daily Lesson',
    parts,
  };
  startLessonView();
}

const noComboMistakesModalOverlay = document.getElementById('noComboMistakesModalOverlay');
const noComboMistakesModalBody = document.getElementById('noComboMistakesModalBody');
const noComboMistakesModalCloseBtn = document.getElementById('noComboMistakesModalCloseBtn');
noComboMistakesModalCloseBtn?.addEventListener('click', () => noComboMistakesModalOverlay.classList.remove('show'));

// Entry point called from the Review Page's Daily Lesson button (see
// below) and Home's weekly-review recap. Premium gate first (no XP Shop
// credit fallback here — this is a straight Premium perk), then checks
// there's actually more than one course to combine, then checks there are
// mistakes to combine them around, then checks it hasn't already been
// done today (saved locally only — see getDailyLessonDoneToday()).
export async function openComboLesson() {
  if (!isPremium()) {
    openPaywall({ reason: 'Daily Lesson is a Premium feature — mix all your courses and toughest questions into one.' });
    return;
  }
  if (getDailyLessonDoneToday()) return; // button should already be greyed out — fail safe, don't restart it
  if (courses.length < 2) {
    if (noComboMistakesModalOverlay) {
      noComboMistakesModalBody.textContent = 'Create at least 2 courses to combine them into a Daily Lesson.';
      noComboMistakesModalOverlay.classList.add('show');
    }
    return;
  }
  const allWrong = await collectAllCoursesWrongAnswers();
  if (!allWrong.length) {
    if (noComboMistakesModalOverlay) {
      noComboMistakesModalBody.textContent = "You're all caught up across every course — keep learning and we'll save anything you miss here for your next Daily Lesson.";
      noComboMistakesModalOverlay.classList.add('show');
    }
    return;
  }
  startComboLesson(allWrong);
}

// Same as removeWrongAnswer() but for a specific course rather than always
// activeCourse — needed because Combo Lesson's parts can each come from a
// different course.
async function removeWrongAnswerFromCourse(courseId, docId) {
  try {
    const u = uid();
    if (!u || !courseId) return;
    await deleteDoc(doc(db, 'users', u, 'learnCourses', courseId, 'wrongAnswers', docId));
    if (activeCourse?.id === courseId) {
      wrongAnswers = wrongAnswers.filter((w) => w.id !== docId);
      updateReviewWrongAnswersBtn();
    }
  } catch (err) {
    console.error('Failed to remove wrong answer:', err);
  }
}

// ============================================================
// CREATE COURSE
// ============================================================
learnCreateFirstBtn.addEventListener('click', () => openCreateCourseModal());
addCourseBtn.addEventListener('click', () => openCreateCourseModal());
createCourseCancelBtn.addEventListener('click', () => createCourseModalOverlay.classList.remove('show'));

// +1 slot from the one-time-ever "Extra Course Slot" XP Shop purchase
// (5→6 free, 10→11 premium) — stacks with, doesn't replace, the base limit.
function effectiveMaxCourses() {
  return limits().maxCourses + (learnProfile?.extraCourseSlotBought ? 1 : 0);
}

// A purchased "1 Day of Premium" bans course CREATION on the day it's
// active (per spec) even though every other premium perk applies that
// day via hasBankedPremiumPerks() — so this is checked in addition to,
// not instead of, the normal course-count limit below.
function isCourseCreationBlockedByBankedDay() {
  return !isPremium() && hasBankedPremiumPerks(learnProfile) && courses.length >= limits().maxCourses;
}

function openCreateCourseModal(prefillTopic) {
  if (isCourseCreationBlockedByBankedDay()) {
    coursesModalOverlay.classList.remove('show');
    openPaywall({ reason: "Today's a Premium Day, but course creation isn't included — try again tomorrow or upgrade." });
    return;
  }
  if (courses.length >= effectiveMaxCourses()) {
    coursesModalOverlay.classList.remove('show');
    openPaywall({ reason: 'You reached the course limit. Upgrade to continue' });
    return;
  }
  createCourseError.textContent = '';
  createCourseInput.value = prefillTopic || '';
  coursesModalOverlay.classList.remove('show');
  createCourseModalOverlay.classList.add('show');
}

createCourseSubmitBtn.addEventListener('click', async () => {
  const prompt = createCourseInput.value.trim();
  if (!prompt || isCourseCreationBlockedByBankedDay() || courses.length >= effectiveMaxCourses()) return;

  createCourseSubmitBtn.disabled = true;
  createCourseSubmitBtn.textContent = 'Creating…';
  createCourseError.textContent = '';

  try {
    const res = await fetch(LEARN_WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'generateCourse', prompt }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Could not create that course.');

    const u = uid();
    const courseRef = doc(collection(db, 'users', u, 'learnCourses'));
    const courseData = {
      title: data.title,
      description: data.description,
      color: pickRandomCourseColor(),
      prompt,
      createdAt: serverTimestamp(),
      lastOpenedAt: serverTimestamp(),
      status: 'generating', // first lesson is being prepared
      currentUnitIndex: 0,
      currentLessonIndex: 0,
      courseReviewCompleted: false,
      units: data.units.map((u2) => ({ title: u2.title, description: u2.description })),
    };
    await setDoc(courseRef, courseData);

    courses.unshift({ id: courseRef.id, ...courseData });
    activeCourse = courses[0];
    checkCourseBadges(courses.length);

    createCourseModalOverlay.classList.remove('show');
    renderCourseHomeOrEmpty();
    maybeShowOverlay('newCourseCreated', {
      courseColor: courseData.color,
      vars: { courseName: courseData.title },
    });

    // Only the very first lesson gets made right away, so there's something
    // to jump into. Every other lesson is generated on demand, the moment
    // the learner presses Start on it.
    prepareFirstLesson(courseRef.id);
  } catch (err) {
    createCourseError.textContent = err.message || 'Something went wrong. Try again.';
  } finally {
    createCourseSubmitBtn.disabled = false;
    createCourseSubmitBtn.textContent = 'Create Course';
  }
});

// ============================================================
// "I DON'T KNOW WHAT TO LEARN" QUIZ
// ============================================================
// 5-question, free-text quiz (random draw from a 30-question pool each
// time) whose answers get sent to the worker so the AI can propose a
// course topic. On success, the suggested topic is dropped into the
// regular Create Course input so the learner can still review/edit it
// before actually creating the course — this deliberately reuses the
// existing generateCourse flow rather than creating a course directly.
idkWhatToLearnBtn.addEventListener('click', openIdkQuiz);
idkQuizCancelBtn.addEventListener('click', closeIdkQuiz);
idkQuizNextBtn.addEventListener('click', handleIdkQuizNext);
idkQuizInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') handleIdkQuizNext();
});

function openIdkQuiz() {
  idkQuizQuestions = shuffledCopy(IDK_QUIZ_QUESTIONS).slice(0, IDK_QUIZ_LENGTH);
  idkQuizIndex = 0;
  idkQuizAnswers = [];
  idkQuizLoading.classList.remove('show');
  idkQuizBody.style.display = '';
  idkQuizNextBtn.disabled = false;
  renderIdkQuizStep();
  createCourseModalOverlay.classList.remove('show');
  idkQuizModalOverlay.classList.add('show');
}

function closeIdkQuiz() {
  idkQuizModalOverlay.classList.remove('show');
}

function renderIdkQuizStep() {
  idkQuizError.textContent = '';
  idkQuizInput.value = '';
  idkQuizProgress.textContent = `Question ${idkQuizIndex + 1} of ${IDK_QUIZ_LENGTH}`;
  idkQuizQuestion.textContent = idkQuizQuestions[idkQuizIndex];
  idkQuizNextBtn.textContent = idkQuizIndex === IDK_QUIZ_LENGTH - 1 ? 'Get My Topic' : 'Next';

  idkQuizDots.innerHTML = idkQuizQuestions.map((_, i) => `
    <div class="idk-quiz-dot${i < idkQuizIndex ? ' filled' : ''}"></div>
  `).join('');

  idkQuizInput.focus();
}

async function handleIdkQuizNext() {
  const answer = idkQuizInput.value.trim();
  if (!answer) {
    idkQuizError.textContent = 'Type an answer to continue.';
    return;
  }
  idkQuizAnswers.push({ question: idkQuizQuestions[idkQuizIndex], answer });

  if (idkQuizIndex < IDK_QUIZ_LENGTH - 1) {
    idkQuizIndex += 1;
    renderIdkQuizStep();
    return;
  }

  await submitIdkQuiz();
}

async function submitIdkQuiz() {
  idkQuizBody.style.display = 'none';
  idkQuizLoading.classList.add('show');
  idkQuizNextBtn.disabled = true;

  try {
    const res = await fetch(LEARN_WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'suggestCourseTopic', answers: idkQuizAnswers }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Could not come up with a topic.');

    idkQuizModalOverlay.classList.remove('show');
    createCourseError.textContent = '';
    createCourseInput.value = data.topic || '';
    createCourseModalOverlay.classList.add('show');
    createCourseInput.focus();
  } catch (err) {
    idkQuizBody.style.display = '';
    idkQuizLoading.classList.remove('show');
    idkQuizNextBtn.disabled = false;
    idkQuizError.textContent = err.message || 'Something went wrong. Try again.';
  }
}

// Reads topicsCovered off already-generated lesson docs for a course, so the
// AI generator can be told what's already been taught and avoid reteaching it.
// Pass { unitIndex } to restrict to lessons within one unit (used for regular
// lesson generation); omit it to gather topics across the whole course (used
// for the final course review).
async function collectPreviousTopics(courseId, { unitIndex } = {}) {
  const u = uid();
  const lessonsSnap = await getDocs(collection(db, 'users', u, 'learnCourses', courseId, 'lessons'));
  const topics = [];
  lessonsSnap.forEach((docSnap) => {
    if (docSnap.id === 'courseReview') return;
    if (unitIndex !== undefined && !docSnap.id.startsWith(`${unitIndex}_`)) return;
    const data = docSnap.data();
    if (Array.isArray(data.topicsCovered)) topics.push(...data.topicsCovered);
  });
  return topics;
}

async function prepareFirstLesson(courseId) {
  const u = uid();
  const courseRef = doc(db, 'users', u, 'learnCourses', courseId);
  try {
    const courseSnap = await getDoc(courseRef);
    if (!courseSnap.exists()) return;
    const course = courseSnap.data();
    await generateAndSaveLesson(courseId, course.title, course.description, course.units[0], 0, 0);
  } catch (err) {
    console.error('First lesson generation failed:', err);
  } finally {
    await updateDoc(courseRef, { status: 'ready' });
    if (activeCourse?.id === courseId) {
      activeCourse.status = 'ready';
      renderCourseHomeOrEmpty({ resetView: false });
    }
  }
}

// Calls the worker for a single lesson (or the unit review, the 15th lesson
// in every unit) and saves it to Firestore. Shared by first-lesson prep and
// on-demand generation from the Start button. If a title was already
// pre-generated (see fetchAndSaveLessonTitle), pass it along so Groq writes
// content to match it instead of inventing a new one, and it's used as the
// saved title verbatim rather than whatever the model echoes back.
// Same localStorage flag main.js's Settings > Additional Settings >
// "Diagrams in Lessons" switch writes (see getDiagramsEnabled() there) —
// learn.js and main.js can't import each other directly, so this reads the
// raw key rather than duplicating a shared module. Used both to gate
// whether a lesson part ever renders a diagram (see diagramHtml() further
// below) and, here, to skip asking the worker to generate diagrams at all
// when the learner has the feature off, saving the extra generation call.
function getDiagramsEnabledFlag() {
  try { return localStorage.getItem('kll_diagrams_enabled') === '1'; } catch { return false; }
}

// Same "on by default" localStorage flag main.js's Settings > Additional
// Settings > "Adaptive Difficulty" switch writes (getAdaptiveDifficultyEnabled()
// there) — mirrors getDiagramsEnabledFlag() above for the same
// can't-import-main.js-from-learn.js reason. A missing/unset value reads
// as enabled (true).
function getAdaptiveDifficultyEnabledFlag() {
  try {
    const stored = localStorage.getItem('kll_adaptive_difficulty_enabled');
    return stored === null ? true : stored === '1';
  } catch { return true; }
}

// Reads { correct, total } for the last regular lesson/unit-review this
// learner completed in `course` (see finishLesson() below, which writes
// this onto the course doc) and turns it into a difficulty signal for the
// NEXT lesson's generation — 'easier' at <=40% (2/5 or worse), 'harder' at
// >=80% (4/5 or better), otherwise 'next' (steady). Returns null if
// adaptive difficulty is off, or there's no prior score to go on yet
// (first lesson in a course).
function computeDifficultySignal(course) {
  if (!getAdaptiveDifficultyEnabledFlag()) return null;
  const last = course?.lastLessonScore;
  if (!last || !last.total) return null;
  const ratio = last.correct / last.total;
  if (ratio <= 0.4) return 'easier';
  if (ratio >= 0.8) return 'harder';
  return 'next';
}

// ============================================================
// LOCAL LESSON CACHE (precaution only)
// Lesson content and lesson titles are shown to the learner as soon as
// they're generated — the Firestore write happens in the background
// afterward, not before. If the learner leaves the app (closes the tab,
// loses connectivity, etc.) before that background write lands, this is
// what's used to recover it on return. Each save is cleared the moment its
// real Firestore write confirms, so under normal conditions these never
// stick around — they're a bridge over the gap between "shown" and
// "persisted", not a second source of truth.
function localLessonKey(courseId, unitIndex, lessonIndex) {
  return `kll_local_lesson_${uid()}_${courseId}_${unitIndex}_${lessonIndex}`;
}
function saveLessonLocally(courseId, unitIndex, lessonIndex, data) {
  try { localStorage.setItem(localLessonKey(courseId, unitIndex, lessonIndex), JSON.stringify(data)); } catch {}
}
function getLocalLesson(courseId, unitIndex, lessonIndex) {
  try {
    const raw = localStorage.getItem(localLessonKey(courseId, unitIndex, lessonIndex));
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}
function clearLocalLesson(courseId, unitIndex, lessonIndex) {
  try { localStorage.removeItem(localLessonKey(courseId, unitIndex, lessonIndex)); } catch {}
}

// ---- Recent-completed-lessons cache (last 5) ----
// Separate purpose from the write-ahead saveLessonLocally()/getLocalLesson()
// pair above: those exist only to bridge the gap between "shown" and
// "written to Firestore" for the lesson currently in progress, and get
// cleared the moment that write confirms. This is a rolling history of the
// last 5 lessons the learner has actually FINISHED, kept around after
// completion (e.g. so a "recently completed" list can render instantly
// without a Firestore round trip). Oldest entry drops the moment a 6th is
// added, so this never grows unbounded.
const RECENT_LESSONS_MAX = 5;
function recentLessonsKey() {
  return `kll_recent_lessons_${uid()}`;
}
function getRecentLessons() {
  try {
    const raw = localStorage.getItem(recentLessonsKey());
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch { return []; }
}
// Adds `entry` as the newest recently-completed lesson, evicting the
// oldest one if this pushes the list past RECENT_LESSONS_MAX. `entry`
// should already be plain-serializable (no Firestore refs/Timestamps).
function pushRecentLesson(entry) {
  try {
    const list = getRecentLessons();
    list.unshift(entry);
    while (list.length > RECENT_LESSONS_MAX) list.pop(); // drop oldest
    localStorage.setItem(recentLessonsKey(), JSON.stringify(list));
  } catch (err) {
    console.warn('Could not update recent-lessons cache:', err);
  }
}
export { getRecentLessons };

// Calls the worker for a single lesson (or the unit review, the 15th lesson
// in every unit) and returns the lesson data — does NOT write it to
// Firestore. Split out from generateAndSaveLesson() below so the on-demand
// Start-button flow can show the lesson the moment this resolves and save
// to Firestore afterward in the background, while flows that don't have a
// learner waiting on-screen (course prep) can still generate-and-save in
// one step via generateAndSaveLesson().
async function generateLessonData(courseId, courseTitle, courseDescription, unit, unitIndex, lessonIndex, pregeneratedTitle) {
  const isReview = lessonIndex === LESSONS_PER_UNIT - 1;
  const previousTopics = await collectPreviousTopics(courseId, { unitIndex });

  const enableVisuals = getDiagramsEnabledFlag();
  const difficulty = computeDifficultySignal(activeCourse?.id === courseId ? activeCourse : null);

  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      action: isReview ? 'generateUnitReview' : 'generateLesson',
      courseTitle,
      courseDescription,
      unitTitle: unit.title,
      unitDescription: unit.description,
      lessonNumber: lessonIndex + 1,
      previousTopics,
      lessonTitle: !isReview ? (pregeneratedTitle || undefined) : undefined,
      enableVisuals, // <-- Include the toggle state here
      difficulty: difficulty || undefined,
    }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Lesson generation failed.');

  return {
    lessonTitle: (!isReview && pregeneratedTitle) ? pregeneratedTitle : data.lessonTitle,
    parts: data.parts,
    topicsCovered: Array.isArray(data.topicsCovered) ? data.topicsCovered : [],
    isUnitReview: isReview,
    isCourseReview: false,
    status: 'available',
    createdAt: serverTimestamp(),
  };
}

// Generates AND saves (awaiting the Firestore write) in one step. Used by
// flows with no learner staring at a loading state — e.g. course-prep's
// first-lesson pregeneration — where there's no UI benefit to returning
// early and the simplicity of "done means saved" is worth more than it is
// on the interactive Start-button path below.
async function generateAndSaveLesson(courseId, courseTitle, courseDescription, unit, unitIndex, lessonIndex, pregeneratedTitle) {
  const u = uid();
  const lessonData = await generateLessonData(courseId, courseTitle, courseDescription, unit, unitIndex, lessonIndex, pregeneratedTitle);
  const lessonRef = doc(db, 'users', u, 'learnCourses', courseId, 'lessons', `${unitIndex}_${lessonIndex}`);
  await setDoc(lessonRef, lessonData);
  return lessonRef;
}

// ============================================================
// COURSE REVIEW (final 15-question review, generated on demand)
// ============================================================
courseReviewErrorCloseBtn.addEventListener('click', () => courseReviewErrorModalOverlay.classList.remove('show'));

learnCourseReviewBtn.addEventListener('click', async () => {
  const u = uid();
  const reviewRef = doc(db, 'users', u, 'learnCourses', activeCourse.id, 'lessons', 'courseReview');
  const snap = await getDoc(reviewRef);
  if (snap.exists()) {
    currentLesson = { id: 'courseReview', ref: reviewRef, isCourseReview: true, ...snap.data() };
    startLessonView();
    return;
  }

  learnCourseReviewBtn.disabled = true;
  learnCourseReviewBtn.textContent = 'Creating your review…';
  try {
    const courseSnap = await getDoc(doc(db, 'users', u, 'learnCourses', activeCourse.id));
    const course = courseSnap.data();
    const previousTopics = await collectPreviousTopics(activeCourse.id);
    const res = await fetch(LEARN_WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'generateCourseReview',
        courseTitle: course.title,
        courseDescription: course.description,
        unitTitles: course.units.map((un) => un.title),
        previousTopics,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Could not create the review.');

    await setDoc(reviewRef, {
      lessonTitle: data.lessonTitle,
      parts: data.parts,
      isCourseReview: true,
      status: 'available',
    });

    const freshSnap = await getDoc(reviewRef);
    currentLesson = { id: 'courseReview', ref: reviewRef, isCourseReview: true, ...freshSnap.data() };
    startLessonView();
  } catch (err) {
    console.error('Course review generation failed:', err);
    courseReviewErrorModalOverlay.classList.add('show');
  } finally {
    learnCourseReviewBtn.disabled = false;
    renderCourseHomeOrEmpty();
  }
});

// ============================================================
// COURSES MODAL (switch between / delete courses)
// ============================================================
learnCoursesBtn.addEventListener('click', () => {
  renderCoursesList();
  coursesModalOverlay.classList.add('show');
});
coursesModalCloseBtn.addEventListener('click', () => coursesModalOverlay.classList.remove('show'));

function renderCoursesList() {
  coursesList.innerHTML = courses.map((c) => {
    const unitsDone = Math.min(c.currentUnitIndex || 0, UNITS_PER_COURSE);
    const isDone = unitsDone >= UNITS_PER_COURSE;
    const pct = Math.round((unitsDone / UNITS_PER_COURSE) * 100);
    const progressLabel = isDone ? 'Completed' : `Unit ${unitsDone + 1} of ${UNITS_PER_COURSE}`;
    return `
    <div class="course-list-item${activeCourse?.id === c.id ? ' active-course' : ''}" style="--item-color:${c.color}">
      <button class="course-item-main" data-id="${c.id}">
        <div class="course-item-title">${escapeHtml(c.title)}</div>
        <div class="course-item-desc">${escapeHtml(c.description)}</div>
        <div class="course-item-progress-row">
          <div class="course-item-progress-bar"><div class="course-item-progress-fill" style="width:${pct}%"></div></div>
          <div class="course-item-progress-label">${progressLabel}</div>
        </div>
      </button>
      <button class="course-item-share-btn" data-id="${c.id}" aria-label="Share course">
        <span class="material-symbols-outlined">ios_share</span>
      </button>
      <button class="course-item-delete-btn" data-id="${c.id}" aria-label="Delete course">
        <span class="material-symbols-outlined">delete</span>
      </button>
    </div>
  `;
  }).join('');

  coursesList.querySelectorAll('.course-item-share-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const c = courses.find((x) => x.id === btn.dataset.id);
      if (c) openShareCourseModal(c);
    });
  });

  coursesList.querySelectorAll('.course-item-main').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const c = courses.find((x) => x.id === btn.dataset.id);
      if (!c) return;
      activeCourse = c;
      coursesModalOverlay.classList.remove('show');
      renderCourseHomeOrEmpty();
      const u = uid();
      await updateDoc(doc(db, 'users', u, 'learnCourses', c.id), { lastOpenedAt: serverTimestamp() });
    });
  });

  coursesList.querySelectorAll('.course-item-delete-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const c = courses.find((x) => x.id === btn.dataset.id);
      if (c) openDeleteCourseModal(c);
    });
  });
}
let deleteCourseTarget = null;
const track = document.getElementById('deleteCourseSlideTrack');
const thumb = document.getElementById('deleteCourseSlideThumb');
const slideFill = document.getElementById('deleteCourseSlideFill');
const errorDiv = document.getElementById('deleteCourseError');

function openDeleteCourseModal(course) {
  deleteCourseTarget = course;
  deleteCourseModalTitle.textContent = `"${course.title}"`;
  errorDiv.textContent = '';
  resetSlider();
  deleteCourseModalOverlay.classList.add('show');
}

function closeDeleteCourseModal() {
  deleteCourseModalOverlay.classList.remove('show');
  deleteCourseTarget = null;
  resetSlider();
}

function resetSlider() {
  thumb.style.transform = 'translateX(0px)';
  if (slideFill) slideFill.style.width = '44px';
}

// Drag handling via Pointer Events (not mouse events) — this modal runs
// inside a Capacitor WKWebView on iOS, where mousedown/mousemove/mouseup
// don't reliably fire for touch input, which is why "Slide right to
// delete" previously did nothing on device despite working with a mouse
// in a desktop browser. Pointer Events unify mouse/touch/pen and, with
// setPointerCapture, keep tracking the drag even if the finger moves
// outside the thumb's bounds mid-swipe.
let isDragging = false;
let startX = 0;

thumb.style.touchAction = 'none';

thumb.addEventListener('pointerdown', (e) => {
  isDragging = true;
  startX = e.clientX;
  errorDiv.textContent = '';
  thumb.setPointerCapture?.(e.pointerId);
});

thumb.addEventListener('pointermove', (e) => {
  if (!isDragging) return;
  const maxX = track.clientWidth - thumb.clientWidth - 8;
  const deltaX = e.clientX - startX;
  const currentX = Math.max(0, Math.min(deltaX, maxX));

  thumb.style.transform = `translateX(${currentX}px)`;
  if (slideFill) slideFill.style.width = `${44 + currentX}px`;

  // Check if dragged to the end (with a small threshold)
  if (currentX >= maxX - 5) {
    isDragging = false;
    thumb.releasePointerCapture?.(e.pointerId);
    handleSuccessfulSlide();
  }
});

function endDrag(e) {
  if (!isDragging) return;
  isDragging = false;
  thumb.releasePointerCapture?.(e.pointerId);
  // Snap back if released before completion
  const matrix = window.getComputedStyle(thumb).transform;
  const currentX = matrix !== 'none' ? parseFloat(matrix.split(',')[4]) : 0;
  const maxX = track.clientWidth - thumb.clientWidth - 8;

  if (currentX < maxX - 5) {
    resetSlider();
  }
}

thumb.addEventListener('pointerup', endDrag);
thumb.addEventListener('pointercancel', endDrag);

async function handleSuccessfulSlide() {
  if (!deleteCourseTarget) return;
  const courseId = deleteCourseTarget.id;
  
  errorDiv.textContent = 'Deleting…';
  await deleteCourse(courseId);
  closeDeleteCourseModal();
}

deleteCourseCancelBtn.addEventListener('click', closeDeleteCourseModal);

async function deleteCourse(courseId) {
  const u = uid();
  try {
    const lessonsSnap = await getDocs(collection(db, 'users', u, 'learnCourses', courseId, 'lessons'));
    await Promise.all(lessonsSnap.docs.map((d) => deleteDoc(d.ref)));
  } catch (err) {
    console.error('Could not clean up lessons for deleted course:', err);
  }
  await deleteDoc(doc(db, 'users', u, 'learnCourses', courseId));

  courses = courses.filter((c) => c.id !== courseId);
  if (activeCourse?.id === courseId) {
    activeCourse = courses[0] || null;
  }
  renderCoursesList();
  renderCourseHomeOrEmpty();
}
// ============================================================
// SHARE COURSE — send a copy of a course to a friend, who sees it at the
// top of their Learn page until they Accept or Decline it.
// ============================================================

// Reads a friend's public profile for display in the friend picker. Mirrors
// fetchMiniProfile() in main.js — kept local since learn.js is a separate module.
async function fetchShareMiniProfile(otherUid) {
  try {
    const snap = await getDoc(doc(db, 'userProfiles', otherUid));
    if (snap.exists()) return snap.data();
  } catch { /* not an accepted friend (shouldn't happen here) — fall through */ }
  try {
    const dirSnap = await getDoc(doc(db, 'userDirectory', otherUid));
    if (dirSnap.exists()) return { email: dirSnap.data().email };
  } catch { /* ignore */ }
  return {};
}

shareCourseHeaderBtn.addEventListener('click', () => {
  if (activeCourse) openShareCourseModal(activeCourse);
});
shareCourseCloseBtn.addEventListener('click', () => shareCourseModalOverlay.classList.remove('show'));

async function openShareCourseModal(course) {
  shareCourseTarget = course;
  shareCourseModalSub.innerHTML = `Send "${course.title}" to one of your friends below OR send it to anyone by scanning their QR code (tell them to open Profile > Friends > My QR Code).`;
  shareCourseError.textContent = '';
  shareCourseFriendsList.innerHTML = '';
  shareCourseEmpty.style.display = 'none';
  coursesModalOverlay.classList.remove('show');
  shareCourseModalOverlay.classList.add('show');

  const u = uid();
  if (!u) return;
  let friends = [];
  try {
    const snap = await getDocs(query(collection(db, 'users', u, 'friends'), where('status', '==', 'accepted')));
    friends = snap.docs.map((d) => ({ uid: d.id, ...d.data() }));
  } catch (err) {
    console.error('Failed to load friends for sharing:', err);
  }

  if (!friends.length) {
    shareCourseEmpty.style.display = '';
    return;
  }

  for (const friend of friends) {
    const info = await fetchShareMiniProfile(friend.uid);
    const row = document.createElement('div');
    row.className = 'share-course-friend-row';
    row.innerHTML = `
      <div>
        <div class="share-course-friend-name">${escapeHtml(info.displayName || info.email || 'Learner')}</div>
        <div class="share-course-friend-sub">${escapeHtml(info.email || '')}</div>
      </div>
      <button type="button" class="share-course-send-btn" data-uid="${friend.uid}">Send</button>
    `;
    row.querySelector('.share-course-send-btn').addEventListener('click', (e) => sendCourseShare(friend.uid, e.currentTarget, info.email || info.displayName));
    shareCourseFriendsList.appendChild(row);
  }
}

// DOM-free core: writes the pending share + notifies the recipient. Shared
// by the friend-list "Send" button below and the QR-scan flow further down
// — both just need "share shareCourseTarget with this uid", they differ
// only in how the target uid was discovered and how progress is displayed.
async function shareCourseCore(toUid) {
  const u = uid();
  if (!u || !shareCourseTarget) throw new Error('Nothing to share.');
  if (toUid === u) throw new Error("That's your own code!");
  const me = auth.currentUser;
  const shareRef = doc(collection(db, 'users', toUid, 'sharedCourses'));
  await setDoc(shareRef, {
    fromUid: u,
    fromName: me?.displayName || (me?.email ? me.email.split('@')[0] : 'A friend'),
    fromEmail: me?.email || '',
    title: shareCourseTarget.title,
    description: shareCourseTarget.description,
    color: shareCourseTarget.color,
    prompt: shareCourseTarget.prompt || null,
    units: shareCourseTarget.units,
    status: 'pending',
    createdAt: serverTimestamp(),
  });
  notifyUser(toUid, {
    type: 'course_shared',
    title: 'A course was shared with you',
    body: `${me?.displayName || (me?.email ? me.email.split('@')[0] : 'A friend')} shared "${shareCourseTarget.title}" with you`,
    data: { fromUid: u, shareId: shareRef.id },
  });
}

async function sendCourseShare(toUid, btn, recipientLabel) {
  btn.disabled = true;
  btn.textContent = 'Sending…';
  shareCourseError.textContent = '';
  try {
    await shareCourseCore(toUid);
    btn.textContent = 'Sent';
    maybeShowOverlay('courseSent', {
      vars: { courseName: shareCourseTarget?.title || '', recipient: recipientLabel || 'your friend' },
    });
  } catch (err) {
    console.error('Failed to share course:', err);
    btn.disabled = false;
    btn.textContent = 'Send';
    shareCourseError.textContent = err.message || 'Could not send that course.';
  }
}

// ---- Share via QR — scan a friend's "My QR Code" (their raw uid, see
// main.js) straight from this modal, no friends-list lookup needed. Uid
// payloads are mixed-case, so preserveCase must be true. Routed through
// identifyScannedCode() rather than assuming the scan is a person code —
// if someone points the camera at a game code by mistake, this jumps
// straight into that game instead of failing to "share" a game code. ----
shareCourseScanBtn?.addEventListener('click', async () => {
  if (!shareCourseTarget) return;
  shareCourseError.textContent = '';
  if (shareCourseScanStatus) shareCourseScanStatus.textContent = '';
  shareCourseScanBtn.disabled = true;
  try {
    const scanned = await scanJoinCode({
      statusText: "Point the camera at your friend's QR code",
      preserveCase: true,
    });
    if (!scanned) return; // user canceled

    await maybeShowOverlay('qrProcessing');
    const result = await identifyScannedCode(scanned);

    if (result.type === 'game') {
      // Not a person's code — a live game session. Jump into it instead.
      shareCourseModalOverlay.classList.remove('show');
      coursesModalOverlay.classList.remove('show');
      try {
        await joinGameByCode(result.code);
      } catch (err) {
        shareCourseModalOverlay.classList.add('show');
        shareCourseError.textContent = err.message || "Couldn't join that game.";
      }
      return;
    }

    if (result.type === 'unknown') {
      shareCourseError.textContent = "That code wasn't recognized.";
      return;
    }

    await shareCourseCore(result.uid);
    if (shareCourseScanStatus) shareCourseScanStatus.textContent = 'Course sent!';
  } catch (err) {
    console.error('Failed to share course via QR:', err);
    shareCourseError.textContent = err.message || 'Could not send that course.';
  } finally {
    shareCourseScanBtn.disabled = false;
  }
});

// ---- Incoming shares: live banner at the top of the Learn page ----
function startSharedCoursesListener() {
  const u = uid();
  if (!u) return;
  stopSharedCoursesListener();
  const q = query(collection(db, 'users', u, 'sharedCourses'), where('status', '==', 'pending'));
  unsubscribeSharedCourses = onSnapshot(q, (snap) => {
    pendingShares = snap.docs.map((d) => ({ id: d.id, ...d.data() }));
    renderSharedCoursesBanner();
  }, (err) => console.error('Shared-courses listener failed:', err));
}

function stopSharedCoursesListener() {
  if (unsubscribeSharedCourses) {
    unsubscribeSharedCourses();
    unsubscribeSharedCourses = null;
  }
}

function renderSharedCoursesBanner() {
  if (!learnSharedCoursesBanner) return;
  if (!pendingShares.length) {
    learnSharedCoursesBanner.style.display = 'none';
    learnSharedCoursesList.innerHTML = '';
    return;
  }
  learnSharedCoursesBanner.style.display = 'block';
  const sorted = [...pendingShares].sort((a, b) => (b.createdAt?.toMillis?.() || 0) - (a.createdAt?.toMillis?.() || 0));
  learnSharedCoursesList.innerHTML = sorted.map((s) => `
    <div class="shared-course-card" style="--item-color:${s.color || '#1E6FE0'}">
      <div class="shared-course-info">
        <div class="shared-course-from">${escapeHtml(s.fromName || s.fromEmail || 'A friend')} shared a course</div>
        <div class="shared-course-title">${escapeHtml(s.title)}</div>
        <div class="shared-course-desc">${escapeHtml(s.description || '')}</div>
      </div>
      <div class="shared-course-btns">
        <button type="button" class="shared-course-btn accept" data-id="${s.id}">Accept</button>
        <button type="button" class="shared-course-btn decline" data-id="${s.id}">Decline</button>
      </div>
    </div>
  `).join('');

  learnSharedCoursesList.querySelectorAll('.shared-course-btn.accept').forEach((btn) => {
    btn.addEventListener('click', () => acceptSharedCourse(btn.dataset.id, btn));
  });
  learnSharedCoursesList.querySelectorAll('.shared-course-btn.decline').forEach((btn) => {
    btn.addEventListener('click', () => declineSharedCourse(btn.dataset.id, btn));
  });
}

function showSharedCourseCardError(shareId, message) {
  const btn = learnSharedCoursesList.querySelector(`.shared-course-btn[data-id="${shareId}"]`);
  const card = btn?.closest('.shared-course-card');
  if (!card) return;
  let errEl = card.querySelector('.shared-course-error');
  if (!errEl) {
    errEl = document.createElement('div');
    errEl.className = 'shared-course-error';
    card.appendChild(errEl);
  }
  errEl.textContent = message;
}

async function acceptSharedCourse(shareId, btn) {
  const u = uid();
  const share = pendingShares.find((s) => s.id === shareId);
  if (!u || !share) return;

  if (courses.length >= effectiveMaxCourses()) {
    openPaywall({ reason: 'You reached the course limit. Upgrade to continue' });
    return;
  }

  const btnsRow = btn.closest('.shared-course-btns');
  const rowBtns = btnsRow ? btnsRow.querySelectorAll('.shared-course-btn') : [btn];
  rowBtns.forEach((b) => { b.disabled = true; });

  try {
    const courseRef = doc(collection(db, 'users', u, 'learnCourses'));
    const courseData = {
      title: share.title,
      description: share.description,
      color: share.color || pickRandomCourseColor(),
      prompt: share.prompt || null,
      createdAt: serverTimestamp(),
      lastOpenedAt: serverTimestamp(),
      status: 'generating',
      currentUnitIndex: 0,
      currentLessonIndex: 0,
      courseReviewCompleted: false,
      units: share.units,
    };
    await setDoc(courseRef, courseData);

    courses.unshift({ id: courseRef.id, ...courseData });
    activeCourse = courses[0];
    renderCourseHomeOrEmpty();

    await deleteDoc(doc(db, 'users', u, 'sharedCourses', shareId));
    prepareFirstLesson(courseRef.id);
  } catch (err) {
    console.error('Failed to accept shared course:', err);
    showSharedCourseCardError(shareId, 'Something went wrong. Try again.');
    rowBtns.forEach((b) => { b.disabled = false; });
  }
}

async function declineSharedCourse(shareId, btn) {
  const u = uid();
  if (!u) return;
  const btnsRow = btn?.closest('.shared-course-btns');
  const rowBtns = btnsRow ? btnsRow.querySelectorAll('.shared-course-btn') : [];
  rowBtns.forEach((b) => { b.disabled = true; });
  try {
    await deleteDoc(doc(db, 'users', u, 'sharedCourses', shareId));
  } catch (err) {
    console.error('Failed to decline shared course:', err);
    showSharedCourseCardError(shareId, 'Something went wrong. Try again.');
    rowBtns.forEach((b) => { b.disabled = false; });
  }
}

// ============================================================
// LESSON START MODAL
// ============================================================
// Renders the small "EASIER LESSON" / "HARDER LESSON" / "NEXT LESSON" line
// under the lesson title on the Lesson Start modal, based on the SAME
// computeDifficultySignal() the actual generation call uses — so what the
// learner sees here always matches what they're about to get. Not shown
// for unit reviews (fixed title/content, not part of the per-lesson
// difficulty progression) or when there's no signal yet (feature off, or
// no prior lesson score on this course).
function renderLessonDifficultyBadge(isReview, isPrevious) {
  let badgeEl = document.getElementById('lessonDifficultyBadge');
  if (!badgeEl) {
    badgeEl = document.createElement('div');
    badgeEl.id = 'lessonDifficultyBadge';
    badgeEl.style.cssText = 'display:flex;align-items:center;justify-content:center;gap:4px;font-weight:700;font-size:0.8rem;letter-spacing:0.02em;margin-top:4px;text-align:center;width:100%;';
    lessonStartTitle.insertAdjacentElement('afterend', badgeEl);
  }

  // A lesson before the learner's current frontier is a replay of already-
  // completed material — the adaptive difficulty signal only ever applies
  // to the upcoming lesson, so show a plain "PREVIOUS LESSON" marker with a
  // left-pointing arrow instead of a stale/misleading Easier/Harder/Next badge.
  if (isPrevious && !isReview) {
    const courseColor = activeCourse?.color || '#1E6FE0';
    badgeEl.style.display = 'flex';
    badgeEl.style.color = courseColor;
    badgeEl.innerHTML = `<span class="material-symbols-outlined" style="font-size:1rem;color:${courseColor};">arrow_back</span>PREVIOUS LESSON`;
    return;
  }

  const signal = isReview ? null : computeDifficultySignal(activeCourse);
  if (!signal) {
    badgeEl.style.display = 'none';
    return;
  }

  const courseColor = activeCourse?.color || '#1E6FE0';
  const byType = {
    easier: { color: '#2FAE66', icon: 'bolt', label: 'EASIER LESSON' },
    harder: { color: '#FF8A2B', icon: 'fitness_center', label: 'HARDER LESSON' },
    next: { color: courseColor, icon: 'arrow_forward', label: 'NEXT LESSON' },
  };
  const { color, icon, label } = byType[signal];
  badgeEl.style.display = 'flex';
  badgeEl.style.color = color;
  badgeEl.innerHTML = `<span class="material-symbols-outlined" style="font-size:1rem;color:${color};">${icon}</span>${label}`;
}

async function openLessonStartModal(unitIndex, lessonIndex) {
  pendingLessonRef = { unitIndex, lessonIndex };
  const isReview = lessonIndex === LESSONS_PER_UNIT - 1;
  // "Previous" = strictly before the learner's current frontier lesson in
  // this unit (i.e. already completed, being replayed) — not the lesson
  // they're actually due to take next, which is where Easier/Harder/Next
  // applies.
  const isPreviousLesson = !isReview && (
    unitIndex < activeCourse.currentUnitIndex ||
    (unitIndex === activeCourse.currentUnitIndex && lessonIndex < activeCourse.currentLessonIndex)
  );

  lessonStartTitle.textContent = isReview ? 'Unit Review' : `Loading Lesson Title...`;
  lessonStartSub.textContent = isReview ? '10 questions covering this whole unit.' : 'Loading…';
  renderLessonDifficultyBadge(isReview, isPreviousLesson);
  lessonStartModalOverlay.classList.add('show');

  const u = uid();
  const lessonRef = doc(db, 'users', u, 'learnCourses', activeCourse.id, 'lessons', `${unitIndex}_${lessonIndex}`);
  const snap = await getDoc(lessonRef);

  // Bail out quietly if the modal was closed (or a different lesson picked)
  // while we were fetching.
  if (!pendingLessonRef || pendingLessonRef.unitIndex !== unitIndex || pendingLessonRef.lessonIndex !== lessonIndex) return;

  let data = snap.exists() ? snap.data() : null;

  // Firestore has nothing yet, but a background save from an earlier visit
  // may still be sitting in the local cache (e.g. the learner left the app
  // before it landed) — recover it here instead of re-generating from
  // scratch, and retry the write it never finished.
  if (!data) {
    const local = getLocalLesson(activeCourse.id, unitIndex, lessonIndex);
    if (local) {
      data = local;
      setDoc(lessonRef, local)
        .then(() => clearLocalLesson(activeCourse.id, unitIndex, lessonIndex))
        .catch((err) => console.error('Retry of background lesson save failed (kept locally):', err));
    }
  }

  if (data) {
    if (data.lessonTitle) lessonStartTitle.textContent = data.lessonTitle;
    if (!isReview) {
      lessonStartSub.textContent = data.status === 'completed' ? 'Completed — tap Start to do it again.' : '';
    }
    renderLessonDifficultyBadge(isReview, isPreviousLesson);
  } else if (!isReview) {
    lessonStartSub.textContent = "You haven't started this lesson yet.";
    // Kick off a lightweight title-only generation right away, so the real
    // title shows up as soon as it's ready instead of the generic
    // "Lesson N" placeholder — full content still waits for Start.
    fetchAndSaveLessonTitle(unitIndex, lessonIndex).then((title) => {
      if (title && pendingLessonRef && pendingLessonRef.unitIndex === unitIndex && pendingLessonRef.lessonIndex === lessonIndex) {
        lessonStartTitle.textContent = title;
      }
    });
  }
}

// Generates just a lesson's title, ahead of the full content — so tapping
// an unstarted lesson shows a real, specific title instead of a generic
// "Lesson N" placeholder. The title is handed to the caller (and so shown
// on screen) the moment the worker responds; the Firestore write happens
// afterward in the background, not before, so the learner isn't kept
// waiting on a modal for a write they can't see. As a precaution against
// the learner leaving the app before that background write lands, the
// title is stashed in localStorage first and cleared once Firestore
// confirms it. Deduped per lesson so repeated taps (or a tap right before
// Start) don't fire it twice. If the full lesson has already been
// generated by the time this resolves (e.g. the learner hit Start before
// this landed), it backs off rather than clobbering it.
async function fetchAndSaveLessonTitle(unitIndex, lessonIndex) {
  const key = `${unitIndex}_${lessonIndex}`;
  if (titleFetchPromises.has(key)) return titleFetchPromises.get(key);

  const promise = (async () => {
    try {
      const u = uid();
      const courseId = activeCourse.id;
      const lessonRef = doc(db, 'users', u, 'learnCourses', courseId, 'lessons', key);
      const unit = activeCourse.units[unitIndex];
      const previousTopics = await collectPreviousTopics(courseId, { unitIndex });

      const res = await fetch(LEARN_WORKER_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'generateLessonTitle',
          courseTitle: activeCourse.title,
          courseDescription: activeCourse.description,
          unitTitle: unit.title,
          unitDescription: unit.description,
          lessonNumber: lessonIndex + 1,
          previousTopics,
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error || !data.lessonTitle) return null;

      const freshSnap = await getDoc(lessonRef);
      if (freshSnap.exists()) return freshSnap.data().lessonTitle || null; // full lesson beat us to it

      const titleDoc = { lessonTitle: data.lessonTitle, status: 'title-only', createdAt: serverTimestamp() };

      // Precaution first, then fire the real write in the background —
      // the caller already has data.lessonTitle to show without waiting
      // on either of these.
      saveLessonLocally(courseId, unitIndex, lessonIndex, titleDoc);
      setDoc(lessonRef, titleDoc)
        .then(() => clearLocalLesson(courseId, unitIndex, lessonIndex))
        .catch((err) => console.error('Background lesson-title save failed (kept locally):', err));

      return data.lessonTitle;
    } catch (err) {
      console.error('Lesson title pre-generation failed:', err);
      return null;
    } finally {
      titleFetchPromises.delete(key);
    }
  })();

  titleFetchPromises.set(key, promise);
  return promise;
}
lessonStartCancelBtn.addEventListener('click', () => {
  if (activeLessonGenToken) activeLessonGenToken.cancelled = true;
  pendingLessonRef = null;
  lessonStartModalOverlay.classList.remove('show');
  lessonStartBtn.disabled = false;
  lessonStartBtnLabel.textContent = 'Start';
  stopLessonProgressBar();
});

// Fake progress bar over the Start button while a lesson generates. It has
// no idea how far along Groq actually is — it just fills linearly over 20s
// to give a sense of motion. If generation is still going after 20s, we
// stop pretending to know the ETA and switch the label to a "still
// thinking" message instead, while leaving the bar full.
let lessonProgressTimeout = null;

function startLessonProgressBar() {
  if (!lessonStartBtnFill || !lessonStartBtnLabel) return;
  lessonStartBtnFill.style.transition = 'none';
  lessonStartBtnFill.style.width = '0%';
  // Force a reflow so the width reset above is applied before the
  // transition kicks in below — otherwise the browser can coalesce both
  // style changes and skip straight to the animated state.
  void lessonStartBtnFill.offsetWidth;
  lessonStartBtnFill.style.transition = 'width 20s linear';
  lessonStartBtnFill.style.width = '100%';

  lessonProgressTimeout = setTimeout(() => {
    lessonStartBtnLabel.textContent = "We're just thinking a little bit more…";
  }, 20000);
}

function stopLessonProgressBar() {
  if (lessonProgressTimeout) {
    clearTimeout(lessonProgressTimeout);
    lessonProgressTimeout = null;
  }
  if (lessonStartBtnFill) {
    lessonStartBtnFill.style.transition = 'none';
    lessonStartBtnFill.style.width = '0%';
  }
}

// ============================================================
// "ENABLE AI DIAGRAMS?" INFO MODAL — shown every time the Lesson Start
// button is pressed (before the lesson actually opens), UNLESS diagrams
// are already enabled (kll_diagrams_enabled, the real Settings switch in
// main.js) — a learner who's already opted in never needs to be asked
// again. For learners who haven't opted in, two real choices: "Enable
// Diagrams" flips the same Settings switch on and proceeds, or "No,
// thank you" just proceeds without changing anything (they'll be asked
// again next lesson, same as before).
// ============================================================
const DIAGRAMS_ENABLED_KEY = 'kll_diagrams_enabled';
const diagramsInfoModalOverlay = document.getElementById('diagramsInfoModalOverlay');
const diagramsInfoEnableBtn = document.getElementById('diagramsInfoEnableBtn');
const diagramsInfoDeclineBtn = document.getElementById('diagramsInfoDeclineBtn');

function getDiagramsEnabled() {
  try { return localStorage.getItem(DIAGRAMS_ENABLED_KEY) === '1'; } catch { return false; }
}
function setDiagramsEnabledFromInfoModal(enabled) {
  try { localStorage.setItem(DIAGRAMS_ENABLED_KEY, enabled ? '1' : '0'); } catch {}
  // Keep the real Settings toggle (main.js, Profile > Additional Settings)
  // in sync immediately, in case it's rendered/visible right now.
  try {
    const settingsSwitch = document.getElementById('diagramsToggleSwitch');
    settingsSwitch?.classList.toggle('on', enabled);
    settingsSwitch?.setAttribute('aria-checked', String(enabled));
  } catch { /* Settings switch not on screen — nothing to sync */ }
}

// Shows the info modal (only if diagrams aren't already enabled) and
// resolves once the learner has made a choice.
function maybeShowDiagramsInfoModal() {
  return new Promise((resolve) => {
    if (getDiagramsEnabled() || !diagramsInfoModalOverlay || !diagramsInfoEnableBtn || !diagramsInfoDeclineBtn) {
      resolve();
      return;
    }
    diagramsInfoModalOverlay.classList.add('show');

    function cleanup() {
      diagramsInfoModalOverlay.classList.remove('show');
      diagramsInfoEnableBtn.removeEventListener('click', onEnable);
      diagramsInfoDeclineBtn.removeEventListener('click', onDecline);
    }
    function onEnable() {
      setDiagramsEnabledFromInfoModal(true);
      cleanup();
      resolve();
    }
    function onDecline() {
      cleanup();
      resolve();
    }
    diagramsInfoEnableBtn.addEventListener('click', onEnable);
    diagramsInfoDeclineBtn.addEventListener('click', onDecline);
  });
}

lessonStartBtn.addEventListener('click', async () => {
  if (!pendingLessonRef) return;
  const { unitIndex, lessonIndex } = pendingLessonRef;
  const u = uid();
  const lessonRef = doc(db, 'users', u, 'learnCourses', activeCourse.id, 'lessons', `${unitIndex}_${lessonIndex}`);

  // A generation-cancellation token: cancel taps this specific object's
  // `cancelled` flag rather than trying to abort the in-flight fetch/
  // Firestore write outright (which could leave a half-written lesson doc
  // behind) — everything below just checks it before touching the DOM or
  // committing state, so a cancelled generation's result is quietly
  // discarded whenever it eventually resolves.
  const genToken = { cancelled: false };
  activeLessonGenToken = genToken;

  lessonStartBtn.disabled = true;
  // Deliberately NOT disabling lessonStartCancelBtn — the learner must
  // always be able to back out of lesson generation (or any of the
  // Firestore writes it triggers), not just before it starts.
  lessonStartBtnLabel.textContent = 'Loading lesson…';

  // If the title-only pre-generation for this lesson is still in flight,
  // let it land first so it can't race the full generation below and
  // overwrite it with just a title.
  const key = `${unitIndex}_${lessonIndex}`;
  if (titleFetchPromises.has(key)) {
    await titleFetchPromises.get(key);
  }
  if (genToken.cancelled) return;

  let snap = await getDoc(lessonRef);
  if (genToken.cancelled) return;
  let lessonData;
  const localExisting = !snap.exists() ? getLocalLesson(activeCourse.id, unitIndex, lessonIndex) : null;
  if (snap.exists() && Array.isArray(snap.data().parts)) {
    lessonData = snap.data();
  } else if (localExisting && Array.isArray(localExisting.parts)) {
    // Firestore doesn't have full content yet, but a background save from
    // an earlier visit is sitting locally (learner left the app before it
    // landed) — reuse it and retry the write it never finished, rather
    // than paying for a fresh generation.
    lessonData = localExisting;
    setDoc(lessonRef, localExisting)
      .then(() => clearLocalLesson(activeCourse.id, unitIndex, lessonIndex))
      .catch((err) => console.error('Retry of background lesson save failed (kept locally):', err));
  } else {
    lessonStartBtnLabel.textContent = 'Loading lesson...';
    startLessonProgressBar();
    try {
      const unit = activeCourse.units[unitIndex];
      const pregeneratedTitle = snap.exists() ? snap.data().lessonTitle : null;
      lessonData = await generateLessonData(activeCourse.id, activeCourse.title, activeCourse.description, unit, unitIndex, lessonIndex, pregeneratedTitle);
      if (genToken.cancelled) { stopLessonProgressBar(); return; }

      // Precaution first (survives the learner leaving the app before the
      // write below lands), then save to Firestore in the background —
      // the lesson opens below as soon as generation resolves, it doesn't
      // wait on this write.
      saveLessonLocally(activeCourse.id, unitIndex, lessonIndex, lessonData);
      setDoc(lessonRef, lessonData)
        .then(() => clearLocalLesson(activeCourse.id, unitIndex, lessonIndex))
        .catch((err) => console.error('Background lesson save failed (kept locally):', err));
    } catch (err) {
      console.error('On-demand lesson generation failed:', err);
      stopLessonProgressBar();
      if (genToken.cancelled) return;
      lessonStartBtn.disabled = false;
      lessonStartBtnLabel.textContent = 'Start';
      lessonStartSub.textContent = "Load failed, just press Start again.";
      return;
    }
    stopLessonProgressBar();
  }

  lessonStartBtn.disabled = false;
  lessonStartBtnLabel.textContent = 'Start';

  lessonStartModalOverlay.classList.remove('show');

  currentLesson = { id: lessonRef.id, unitIndex, lessonIndex, ref: lessonRef, isCourseReview: false, ...lessonData };
  pendingLessonRef = null;

  // Lesson content is generated/loaded and confirmed playable at this
  // point — show the diagrams info modal now, right before the lesson
  // actually opens, rather than before generation even started.
  await maybeShowDiagramsInfoModal();

  startLessonView();
});

// ============================================================
// LESSON VIEW (playing a regular lesson / unit review / course review)
// ============================================================
function startLessonView() {
  currentPartIndex = 0;
  lessonCorrectCount = 0;
  lessonQuestionCount = currentLesson.parts.filter((p) => p.type === 'question').length;
  // Review lessons (Review Page's per-topic/full reviews) earn a flat XP
  // amount per correct answer rather than the normal streak-scaled gain,
  // and start their tracker at 0 rather than the usual base-10 — see
  // checkAnswer()'s isReviewLesson branch.
  lessonXp = (currentLesson.isReviewLesson || currentLesson.isDailyLesson) ? 0 : 10;
  lessonStreakCount = 0;
  lessonAnsweredAny = false;
  lessonAnswerLog = [];
  lessonXpTracker.style.display = (currentLesson.isWrongAnswerReview && !currentLesson.isReviewLesson && !currentLesson.isDailyLesson) ? 'none' : 'flex';
  lessonXpTrackerCount.textContent = lessonXp;
  applyCourseColor(activeCourse.color); // lesson-view-container reads the same --cc var, inherited
  lessonViewTitle.textContent = currentLesson.lessonTitle
    || (currentLesson.isCourseReview ? 'Course Review' : (currentLesson.isUnitReview ? 'Unit Review' : ''));
  lessonViewOverlay.classList.add('show');
  resetAiAssistant();
  renderCurrentPart();
}

// Quitting is only a "confirm" moment once they've actually answered
// something — before that, there's no progress or XP to lose.
lessonExitBtn.addEventListener('click', () => {
  if (lessonAnsweredAny) {
    quitLessonModalOverlay.classList.add('show');
  } else {
    exitLessonView();
  }
});

function exitLessonView() {
  lessonViewOverlay.classList.remove('show');
  currentLesson = null;
  _consumingPurchasedReview = false; // a quit never consumes the XP Shop credit — only a completed review does (see finishLesson)
  resetAiAssistant();
}

quitLessonCancelBtn.addEventListener('click', () => {
  quitLessonModalOverlay.classList.remove('show');
});

quitLessonConfirmBtn.addEventListener('click', () => {
  // XP and lesson completion are only ever persisted in finishLesson(), so
  // simply closing the view here without calling it is what "loses" the
  // progress & XP the confirmation warned about.
  quitLessonModalOverlay.classList.remove('show');
  exitLessonView();
});

function renderCurrentPart() {
  const parts = currentLesson.parts;
  const total = parts.length;
  const part = parts[currentPartIndex];

  const pct = (currentPartIndex / total) * 100;
  lessonProgressFill.style.width = `${pct}%`;

  // Discrete orange/green segments for weak/strong review lessons (Full
  // Weak Spot Review, Full Personalized Review) — one segment per
  // question, lighting up as each is reached, so the bar reads as actual
  // per-question progress instead of a pre-drawn static stripe. Every
  // other lesson type keeps the normal smooth --cc fill.
  if (currentLesson.reviewProgressColorMap && currentLesson.reviewProgressColorMap.length) {
    const map = currentLesson.reviewProgressColorMap;
    lessonProgressTrack.classList.add('rp-segmented');
    lessonProgressSegments.innerHTML = map.map((kind, i) => {
      const lit = i <= currentPartIndex;
      return `<div class="rp-segment${lit ? ' rp-lit' : ''} ${kind === '#2FA84F' ? 'rp-strong' : 'rp-weak'}"></div>`;
    }).join('');
  } else {
    lessonProgressTrack.classList.remove('rp-segmented');
    lessonProgressSegments.innerHTML = '';
  }

  lessonFeedback.textContent = '';
  lessonFeedback.className = 'lesson-feedback';
  selectedChoiceIndex = null;
  answerLocked = false;
  lessonWhyBtn.style.display = 'none';
  missedQuestion = null;

  // A part carries ONE of: a real photo (Pexels imageUrl — checked first,
  // since the worker only sets this when a photo genuinely beats an
  // illustration), a named template (rendered as real HTML/CSS via
  // renderDiagramTemplate()), or a rare raw svg fallback.
  //
  // Gated behind the Profile > Additional Settings > "Diagrams: SVG/Pexels
  // (Beta)" toggle (main.js, kll_diagrams_enabled in localStorage) — off by
  // default, so lesson parts render as plain text/question with no diagram
  // at all until the learner opts in.
  const feedbackBtnHtml = `<button type="button" class="lesson-diagram-feedback-btn" data-diagram-feedback-btn>Report feedback</button>`;

  const diagramHtml = (p) => {
    let diagramsEnabled = false;
    try { diagramsEnabled = localStorage.getItem('kll_diagrams_enabled') === '1'; } catch {}
    if (!diagramsEnabled) { currentDiagramFeedbackContext = null; return ''; }

    const imageUrl = sanitizeLessonImageUrl(p.imageUrl);
    if (imageUrl) {
      const alt = escapeHtml(p.imageAlt || '');
      const credit = escapeHtml(p.imageCredit || 'Pexels');
      currentDiagramFeedbackContext = { kind: 'image', imageUrl, imageAlt: p.imageAlt || '', imageCredit: p.imageCredit || 'Pexels' };
      return `<div class="lesson-image-wrap">
        <img src="${imageUrl}" alt="${alt}" loading="lazy" />
        <div class="lesson-image-credit">Photo by ${credit} on Pexels</div>
      </div>${feedbackBtnHtml}`;
    }
    if (p.template && typeof p.template === 'string') {
      const html = renderDiagramTemplate(p.template, p.templateData);
      if (!html) { currentDiagramFeedbackContext = null; return ''; }
      currentDiagramFeedbackContext = { kind: 'template', template: p.template, templateData: p.templateData };
      return `<div class="lesson-diagram-wrap">${html}</div>${feedbackBtnHtml}`;
    }
    const svg = sanitizeLessonSvg(p.svg);
    if (!svg) { currentDiagramFeedbackContext = null; return ''; }
    currentDiagramFeedbackContext = { kind: 'svg', svgMarkup: svg };
    return `<div class="lesson-svg-wrap">${svg}</div>${feedbackBtnHtml}`;
  };

  if (part.type === 'text') {
    lessonPartContent.innerHTML = `
      ${diagramHtml(part)}
      <div class="lesson-text-part">${escapeHtml(part.content)}</div>
    `;
    wireDiagramFeedbackBtn(part);
    lessonActionBtn.textContent = 'Next';
    lessonActionBtn.disabled = false;
    lessonActionBtn.onclick = advancePart;
  } else {
    lessonPartContent.innerHTML = `
      ${diagramHtml(part)}
      <div class="lesson-question-text">${escapeHtml(part.question)}</div>
      ${part.choices.map((choice, i) => `<button class="lesson-choice" data-index="${i}">${escapeHtml(choice)}</button>`).join('')}
    `;
    wireDiagramFeedbackBtn(part);
    lessonPartContent.querySelectorAll('.lesson-choice').forEach((btn) => {
      btn.addEventListener('click', () => {
        if (answerLocked) return;
        lessonPartContent.querySelectorAll('.lesson-choice').forEach((b) => b.classList.remove('selected'));
        btn.classList.add('selected');
        selectedChoiceIndex = Number(btn.dataset.index);
        lessonActionBtn.disabled = false;
      });
    });
    lessonActionBtn.textContent = 'Check';
    lessonActionBtn.disabled = true;
    lessonActionBtn.onclick = checkAnswer;
  }
}

function checkAnswer() {
  if (answerLocked || selectedChoiceIndex === null) return;
  answerLocked = true;
  const part = currentLesson.parts[currentPartIndex];
  const buttons = lessonPartContent.querySelectorAll('.lesson-choice');

  buttons.forEach((btn, i) => {
    if (i === part.correctIndex) btn.classList.add('correct');
    else if (i === selectedChoiceIndex) btn.classList.add('wrong');
  });

  const isCorrect = selectedChoiceIndex === part.correctIndex;
  lessonAnsweredAny = true;
  if (isCorrect) {
    lessonCorrectCount++;
    playCorrectSound();
    // XP: +20 for a normal correct answer, +30 on hitting exactly 3 in a
    // row (20 base + a one-time 10 streak bonus), then +25 for each further
    // correct answer beyond that, until the streak breaks and resets.
    if (currentLesson.isReviewLesson) {
      // Flat per-question XP for Review Page lessons — no streak scaling.
      const gain = currentLesson.reviewXpPerCorrect || 0;
      lessonXp += gain;
      lessonXpTrackerCount.textContent = lessonXp;
      if (gain > 0) window.KLLAnim?.flyXpPopup({ container: xpPopupLayer, target: lessonXpTracker, amount: gain });
    } else if (currentLesson.isDailyLesson) {
      // Flat +20 XP for Daily Lesson (renamed Combo Lesson) — no streak scaling.
      lessonXp += 20;
      lessonXpTrackerCount.textContent = lessonXp;
      window.KLLAnim?.flyXpPopup({ container: xpPopupLayer, target: lessonXpTracker, amount: 20 });
    } else if (!currentLesson.isWrongAnswerReview) {
      lessonStreakCount++;
      let gain = 20;
      if (lessonStreakCount === 3) gain = 30;
      else if (lessonStreakCount > 3) gain = 25;
      lessonXp += gain;
      lessonXpTrackerCount.textContent = lessonXp;
      window.KLLAnim?.flyXpPopup({ container: xpPopupLayer, target: lessonXpTracker, amount: gain });
    }
    // Got it right this time during a review/Daily Lesson session — clear
    // it from the legacy wrongAnswers bank (keyed off the part actually
    // carrying a _wrongAnswerDocId, not the lesson type, since Daily
    // Lesson is no longer flagged isWrongAnswerReview).
    if (part._wrongAnswerDocId) {
      if (currentLesson.isComboLesson && part._wrongAnswerCourseId) {
        removeWrongAnswerFromCourse(part._wrongAnswerCourseId, part._wrongAnswerDocId);
      } else {
        removeWrongAnswer(part._wrongAnswerDocId);
      }
    }
  } else {
    playWrongSound();
    lessonStreakCount = 0;
    missedQuestion = {
      question: part.question,
      choices: part.choices,
      correctIndex: part.correctIndex,
      selectedIndex: selectedChoiceIndex,
    };
    lessonWhyBtn.style.display = '';
    // Store every missed question (outside of review/Daily-Lesson
    // sessions, where it's already in the bank) so it can resurface later.
    if (!currentLesson.isWrongAnswerReview && !currentLesson.isDailyLesson) {
      saveWrongAnswer(part);
    }
  }
  // Collect this question+answer for end-of-lesson weak/strong-spot
  // generation — regular lessons only (see finishLesson()'s call into
  // maybeGenerateReviewSpots below).
  if (!currentLesson.isWrongAnswerReview && !currentLesson.isDailyLesson) {
    lessonAnswerLog.push({
      question: part.question,
      choices: part.choices,
      correctIndex: part.correctIndex,
      selectedIndex: selectedChoiceIndex,
      isCorrect,
    });
  }
  lessonFeedback.textContent = isCorrect ? 'Correct!' : 'Not quite';
  lessonFeedback.classList.add(isCorrect ? 'correct' : 'wrong');

  lessonActionBtn.textContent = 'Next';
  lessonActionBtn.disabled = false;
  lessonActionBtn.onclick = advancePart;
}

async function advancePart() {
  currentPartIndex++;
  if (currentPartIndex >= currentLesson.parts.length) {
    await finishLesson();
  } else {
    renderCurrentPart();
  }
}

async function finishLesson() {
  // Wrong-answer reviews (legacy) and Review Page lessons aren't backed by
  // a real lesson doc and never touch the daily streak — but Review Page
  // lessons DO award their flat XP (see checkAnswer()'s isReviewLesson
  // branch, accumulated into lessonXp as the lesson was played).
  if (currentLesson.isWrongAnswerReview) {
    if (_consumingPurchasedReview) {
      _consumingPurchasedReview = false;
      const u = uid();
      learnProfile.purchasedReviewCredits = Math.max(0, (learnProfile.purchasedReviewCredits || 0) - 1);
      if (u) {
        await setDoc(doc(db, 'users', u, 'learnProfile', 'main'), {
          purchasedReviewCredits: learnProfile.purchasedReviewCredits,
        }, { merge: true }).catch(() => {});
      }
    }
    if (currentLesson.isReviewLesson && lessonXp > 0) {
      await awardLessonXp();
    }
    showLessonSummary(false);
    return;
  }

  // Daily Lesson (renamed Combo Lesson): earns flat XP and CAN advance the
  // streak, but still isn't backed by a real lesson doc, so no course-doc
  // write happens. "Done today" is saved locally only (per spec), which
  // greys the button out until local midnight.
  if (currentLesson.isDailyLesson) {
    pauseCelebrations();
    const streakExtended = await bumpStreak();
    if (lessonXp > 0) await awardLessonXp();
    setDailyLessonDoneToday();
    await checkLessonBadges();
    showLessonSummary(streakExtended);
    return;
  }
  // Hold off on any badge celebration popping up until the learner has
  // clicked through the XP overlay (and the streak overlay, if earned) —
  // see showLessonSummary()/lessonSummaryCompleteBtn below for where it
  // resumes. bumpStreak() and checkLessonBadges() both award badges, so
  // this has to wrap them, not just the summary display.
  pauseCelebrations();
  // setDoc+merge rather than updateDoc: the lesson's own background save
  // (see lessonStartBtn's handler / fetchAndSaveLessonTitle above) may not
  // have landed yet, and updateDoc throws on a doc that doesn't exist —
  // merge writes this regardless of whether that background save has
  // resolved, and folds cleanly into it either way.
  await setDoc(currentLesson.ref, { status: 'completed', completedAt: serverTimestamp() }, { merge: true });

  // Push this lesson into the rolling last-5-completed cache. Strip `ref`
  // (a Firestore DocumentReference, not JSON-serializable) and store a
  // plain ISO timestamp instead of the serverTimestamp() sentinel used in
  // the write above, since that sentinel only resolves once written.
  if (!currentLesson.isCourseReview) {
    const { ref, ...serializable } = currentLesson;
    pushRecentLesson({
      ...serializable,
      courseId: activeCourse?.id || null,
      courseTitle: activeCourse?.title || null,
      completedAt: new Date().toISOString(),
    });
  }

  // Record this lesson's score onto the course doc for Adaptive Difficulty
  // to read on the NEXT lesson's generation (see computeDifficultySignal()
  // above) — skipped for the course review, which isn't part of the
  // per-lesson difficulty progression.
  if (activeCourse && !currentLesson.isCourseReview && lessonQuestionCount > 0) {
    const lastLessonScore = { correct: lessonCorrectCount, total: lessonQuestionCount };
    try {
      await updateDoc(doc(db, 'users', uid(), 'learnCourses', activeCourse.id), { lastLessonScore });
      activeCourse.lastLessonScore = lastLessonScore;
    } catch (err) {
      console.error('Failed to save lastLessonScore for Adaptive Difficulty:', err);
    }
  }

  const streakExtended = await bumpStreak();
  await awardLessonXp();
  await checkLessonBadges();
  // AI weak/strong-spot generation — fire-and-forget so a slow/failed
  // worker call never delays showing the lesson summary. Skipped entirely
  // when Review Lessons are toggled off in Additional Settings.
  if (getReviewLessonsEnabled() && lessonAnswerLog.length) {
    maybeGenerateReviewSpots([...lessonAnswerLog]).catch((err) => {
      console.warn('Weak/strong spot generation failed:', err);
    });
  }
  showLessonSummary(streakExtended);
}

// Adds this lesson's earned XP to the account-wide total. Only ever called
// from finishLesson() — a mid-lesson quit never reaches here, which is what
// makes quitting "lose" the XP the confirm modal warns about.
async function awardLessonXp() {
  await awardGameXp(lessonXp);
}

// Shared by the lesson-completion flow and every learning game (Maze,
// Seesaw, Meltdown, Duel) — adds `amount` XP to the account-wide total and
// mirrors it into the public profile doc so friends can see it too.
async function awardGameXp(amount) {
  const u = uid();
  if (!u || !amount) return;
  learnProfile.xp = (learnProfile.xp || 0) + amount;
  const ref = doc(db, 'users', u, 'learnProfile', 'main');
  // Atomic increment (not a full-document overwrite) so this can never
  // clobber XP awarded concurrently elsewhere — e.g. badges.js awarding
  // +100 XP for a badge unlock right around the same time.
  await setDoc(ref, { xp: increment(amount) }, { merge: true });
  await setDoc(doc(db, 'userProfiles', u), { xp: increment(amount), streak: learnProfile.streak }, { merge: true }).catch(() => {});
}

// "1m 05s" for anything a minute or over, otherwise just "12s" — used on
// the various game-complete screens.
function formatGameTime(totalSeconds) {
  const s = Math.max(0, Math.round(totalSeconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}m ${String(rem).padStart(2, '0')}s`;
}

// Shows XP only — streak (if extended) and any earned badges are now their
// own overlays, shown one at a time after this one's Continue is tapped
// (see lessonSummaryCompleteBtn below), so a badge celebration can never
// pop up and block/cover this screen before the learner has even seen it.
function showLessonSummary(streakExtended) {
  pendingStreakExtended = !!streakExtended;
  lessonSummaryContainer.style.setProperty('--cc', activeCourse?.color || '#1E6FE0');
  lessonSummaryScore.textContent = `${lessonCorrectCount}/${lessonQuestionCount}`;
  if (currentLesson.isWrongAnswerReview && !currentLesson.isReviewLesson) {
    lessonSummaryXp.style.display = 'none';
  } else {
    lessonSummaryXpCount.textContent = lessonXp;
    lessonSummaryXp.style.display = 'flex';
  }
  lessonSummaryStreak.style.display = 'none';
  lessonViewOverlay.classList.remove('show');
  lessonSummaryModalOverlay.classList.add('show');
  // Small celebratory confetti burst off the score badge every time any
  // lesson finishes (regular, review, Daily Lesson — all funnel through
  // here), once the summary modal is actually visible.
  requestAnimationFrame(() => window.KLLAnim?.confettiBurst(lessonSummaryScore));
}

lessonSummaryCompleteBtn.addEventListener('click', () => {
  lessonSummaryModalOverlay.classList.remove('show');

  // Wrong-answer reviews never pause celebrations or touch the streak —
  // just go straight back to wherever advanceAfterLessonCompletion() sends them.
  if (currentLesson?.isWrongAnswerReview) {
    advanceAfterLessonCompletion();
    return;
  }

  if (pendingStreakExtended) {
    openStreakModalPostLesson();
  } else {
    resumeCelebrations(advanceAfterLessonCompletion);
  }
});

// Shows the streak overlay as step 2 of the lesson-complete sequence (only
// when the streak was actually extended). This is its own full-screen
// overlay (#lessonStreakOverlay) — not the streak-calendar modal used by
// the manual "view my streak" button — so it reads as a celebration screen
// rather than a settings dialog. The streakModalPostLessonFlow flag stays
// around so other post-lesson bookkeeping can check "are we mid-sequence".
function openStreakModalPostLesson() {
  streakModalPostLessonFlow = true;
  lessonStreakContainer.style.setProperty('--cc', activeCourse?.color || '#1E6FE0');
  lessonStreakCountEl.textContent = learnProfile.streak;
  lessonStreakOverlay.classList.add('show');
}

async function advanceAfterLessonCompletion() {
  const u = uid();

  // ---- Wrong-answer review finished: nothing to advance, just return home ----
  if (currentLesson.isWrongAnswerReview) {
    const wasReviewLesson = currentLesson.isReviewLesson;
    currentLesson = null;
    renderCourseHomeOrEmpty({ resetView: false });
    if (wasReviewLesson) openReviewPage();
    return;
  }

  // ---- Daily Lesson finished: nothing to advance, just return home ----
  if (currentLesson.isDailyLesson) {
    currentLesson = null;
    renderCourseHomeOrEmpty({ resetView: false });
    return;
  }

  // ---- Course review finished: mark the whole course complete ----
  if (currentLesson.isCourseReview) {
    const courseRef = doc(db, 'users', u, 'learnCourses', activeCourse.id);
    await updateDoc(courseRef, { courseReviewCompleted: true, status: 'completed' });
    activeCourse.courseReviewCompleted = true;
    activeCourse.status = 'completed';
    await checkCourseCompletionBadges();

    currentLesson = null;
    renderCourseHomeOrEmpty();
    return;
  }

  // ---- Regular lesson / unit review finished ----
  const { unitIndex, lessonIndex } = currentLesson;
  const isUnitReview = lessonIndex === LESSONS_PER_UNIT - 1;
  // Only advance course progress if this was actually the next lesson the
  // learner was due to take. Replaying an earlier/completed lesson just
  // re-marks it complete without moving the frontier.
  const isFrontierLesson = unitIndex === activeCourse.currentUnitIndex && lessonIndex === activeCourse.currentLessonIndex;

  let resetView = true;

  if (isFrontierLesson) {
    const courseRef = doc(db, 'users', u, 'learnCourses', activeCourse.id);
    if (isUnitReview) {
      const nextUnitIndex = unitIndex + 1;
      await checkUnitCompletionBadges();
      if (nextUnitIndex < UNITS_PER_COURSE) {
        await updateDoc(courseRef, { currentUnitIndex: nextUnitIndex, currentLessonIndex: 0 });
        activeCourse.currentUnitIndex = nextUnitIndex;
        activeCourse.currentLessonIndex = 0;
      } else {
        // All 10 units done — course home will now show the course review card.
        await updateDoc(courseRef, { currentUnitIndex: UNITS_PER_COURSE });
        activeCourse.currentUnitIndex = UNITS_PER_COURSE;
      }
      resetView = true; // hop back to the units overview to reveal what's next
    } else {
      await updateDoc(courseRef, { currentLessonIndex: lessonIndex + 1 });
      activeCourse.currentLessonIndex = lessonIndex + 1;
      resetView = false; // stay put in this unit's path
    }
  } else {
    resetView = false; // a replay — stay right where we were
  }

  currentLesson = null;
  renderCourseHomeOrEmpty({ resetView });
}

// ============================================================
// TRIVIA
// ============================================================
// ---- Learning Games full page (currently just lists Trivia) ----
learnGamesBtn.addEventListener('click', () => {
  gamesPageOverlay.classList.add('show');
  maybeShowOverlay('learningGamesWelcome');
});
gamesExitBtn.addEventListener('click', () => {
  gamesPageOverlay.classList.remove('show');
});
gameCardTrivia.addEventListener('click', () => {
  gamesPageOverlay.classList.remove('show');
  triviaVoiceMode = false;
  openTriviaChooseModal();
});
gameCardVoiceTrivia.addEventListener('click', () => {
  gamesPageOverlay.classList.remove('show');
  triviaVoiceMode = true;
  openTriviaChooseModal();
});
gameCardMaze.addEventListener('click', () => {
  gamesPageOverlay.classList.remove('show');
  openMazeChooseModal();
});
gameCardSeesaw.addEventListener('click', () => {
  gamesPageOverlay.classList.remove('show');
  seesawModeModalOverlay.classList.add('show');
});
joinGameEntryBtn.addEventListener('click', () => {
  gamesPageOverlay.classList.remove('show');
  openJoinGameModal();
});

seesawModeCancelBtn.addEventListener('click', () => {
  seesawModeModalOverlay.classList.remove('show');
});
seesawModeSameDeviceBtn.addEventListener('click', () => {
  seesawModeModalOverlay.classList.remove('show');
  seesawMode = 'local';
  openSeesawChooseModal();
});
seesawModeFriendBtn.addEventListener('click', () => {
  seesawModeModalOverlay.classList.remove('show');
  seesawFriendChoiceModalOverlay.classList.add('show');
});

seesawFriendChoiceCancelBtn.addEventListener('click', () => {
  seesawFriendChoiceModalOverlay.classList.remove('show');
});
seesawHostBtn.addEventListener('click', () => {
  seesawFriendChoiceModalOverlay.classList.remove('show');
  seesawMode = 'host';
  openSeesawChooseModal();
});
seesawJoinBtn.addEventListener('click', () => {
  seesawFriendChoiceModalOverlay.classList.remove('show');
  openJoinGameModal();
});

duelModeCancelBtn.addEventListener('click', () => {
  duelModeModalOverlay.classList.remove('show');
});
duelModeSameDeviceBtn.addEventListener('click', () => {
  duelModeModalOverlay.classList.remove('show');
  duelMode = 'local';
  openDuelChooseModal();
});
duelModeFriendBtn.addEventListener('click', () => {
  duelModeModalOverlay.classList.remove('show');
  duelFriendChoiceModalOverlay.classList.add('show');
});

duelFriendChoiceCancelBtn.addEventListener('click', () => {
  duelFriendChoiceModalOverlay.classList.remove('show');
});
duelHostBtn.addEventListener('click', () => {
  duelFriendChoiceModalOverlay.classList.remove('show');
  duelMode = 'host';
  openDuelChooseModal();
});
duelJoinBtn.addEventListener('click', () => {
  duelFriendChoiceModalOverlay.classList.remove('show');
  openJoinGameModal();
});

function openJoinGameModal() {
  joinGameError.textContent = '';
  joinGameCodeInput.value = '';
  joinGameSubmitBtn.disabled = false;
  joinGameSubmitBtn.textContent = 'Join';
  joinGameModalOverlay.classList.add('show');
  setTimeout(() => joinGameCodeInput.focus(), 50);
}
joinGameCancelBtn.addEventListener('click', () => {
  joinGameModalOverlay.classList.remove('show');
});
joinGameCodeInput.addEventListener('input', () => {
  joinGameCodeInput.value = joinGameCodeInput.value.toUpperCase().replace(/[^A-Z0-9]/g, '');
});

// A join code by itself doesn't say which game it's for — the "Join Game"
// entry point is shared. Try each known namespace in turn (GAME_SESSION_PATHS,
// imported from qrRouting.js); the first one where the code actually exists
// wins. Once joined, branch on session.game to route into that game's
// remote-start flow.
async function joinAnyGameSession(code, guestData) {
  let lastErr;
  for (const gamePath of GAME_SESSION_PATHS) {
    try {
      const { session } = await joinSession(gamePath, code, guestData);
      return { gamePath, session };
    } catch (err) {
      lastErr = err;
      // Only keep trying other namespaces on a "code not found" miss —
      // any other error (already started, already ended) means the code
      // DID match here, so surface that error instead of masking it.
      if (!/wasn't found/i.test(err.message || '')) throw err;
    }
  }
  throw lastErr || new Error("That code wasn't found. Check it and try again.");
}

// Joins a known game session by code and launches the right game view.
// Shared by manual code entry (joinGameSubmitBtn below), the Join Game
// scanner, and any other scanner — Friends "Scan to Add" (main.js), Share
// Course's scanner (below) — that detects a scanned code is actually a
// live game code instead of what that screen was expecting (see
// qrRouting.js's identifyScannedCode). Exported so main.js can call it too.
//
// Applies the same daily-games gate regardless of entry point: even a code
// discovered via a "wrong screen" scan still counts against today's limit.
// Resolves normally (having opened the paywall instead) when gated, so
// every caller can just close its own modal on success either way. Throws
// on an actual join failure (code not found, already started/ended).
export async function joinGameByCode(code) {
  const guestName = (auth.currentUser?.displayName || 'Player 2').split(' ')[0];
  const { gamePath, session } = await joinAnyGameSession(code, {
    guestUid: auth.currentUser?.uid || null,
    guestName,
  });

  if (session.game === 'duel') {
    duelMode = 'guest';
    duelSessionCode = code;
    duelMyPlayer = 2;
    duelQuestions = session.questions || [];
    duelLatestSession = session;

    duelUnsubscribe = listenToSession(gamePath, code, onDuelSessionMessage);

    gateAndStartGame(() => startDuelRemoteView(session), 'duel', (session.questions || []).map((q) => q.question));
    return;
  }

  seesawMode = 'guest';
  seesawSessionCode = code;
  seesawMyPlayer = 2;
  seesawQuestions = session.questions || [];
  seesawLatestSession = session;

  seesawUnsubscribe = listenToSession(gamePath, code, onSeesawSessionMessage);

  gateAndStartGame(() => startSeesawRemoteView(session), 'seesaw', (session.questions || []).map((q) => q.question));
}

joinGameScanBtn.addEventListener('click', async () => {
  joinGameError.textContent = '';
  joinGameScanBtn.disabled = true;
  try {
    // preserveCase: true because we don't yet know whether this is a game
    // code or a person's uid — uid payloads are case-sensitive, and
    // preserving case is a no-op for game codes (their charset is
    // uppercase-only already).
    const scanned = await scanJoinCode({ preserveCase: true });
    if (!scanned) return; // user canceled the scan — leave the modal as-is

    const result = await identifyScannedCode(scanned);

    if (result.type === 'person') {
      // Not a game code — this is someone's "My QR Code" from the Friends
      // page. Send a friend request instead of erroring out.
      try {
        const fr = await sendFriendRequestToUid(result.uid);
        joinGameError.textContent = fr.outcome === 'accepted'
          ? "That was a friend code, not a game code — you're now friends!"
          : "That was a friend code, not a game code — friend request sent!";
      } catch (err) {
        joinGameError.textContent = err.message || 'Could not send that friend request.';
      }
      return;
    }

    if (result.type === 'unknown') {
      joinGameError.textContent = "That code wasn't found. Check it and try again.";
      return;
    }

    joinGameCodeInput.value = result.code;
    joinGameSubmitBtn.click(); // reuse the exact same join flow as typed entry
  } catch (err) {
    joinGameError.textContent = err.message || 'Could not open the scanner.';
  } finally {
    joinGameScanBtn.disabled = false;
  }
});

joinGameSubmitBtn.addEventListener('click', async () => {
  const code = joinGameCodeInput.value.trim().toUpperCase();
  if (!code) {
    joinGameError.textContent = 'Type a code first.';
    return;
  }

  joinGameError.textContent = '';
  joinGameSubmitBtn.disabled = true;
  joinGameSubmitBtn.textContent = 'Connecting…';

  try {
    await joinGameByCode(code);
    joinGameModalOverlay.classList.remove('show');
  } catch (err) {
    joinGameError.textContent = err.message || "Couldn't join that game.";
  } finally {
    joinGameSubmitBtn.disabled = false;
    joinGameSubmitBtn.textContent = 'Join';
  }
});
gameCardMeltdown.addEventListener('click', () => {
  gamesPageOverlay.classList.remove('show');
  openMeltdownChooseModal();
});
gameCardWordGrid.addEventListener('click', () => {
  gamesPageOverlay.classList.remove('show');
  openWordGridChooseModal();
});

// ---- Daily game-limit gate ----
// Shared by every game's "Start" button: checks the daily cap, bumps usage
// on success, and opens the paywall instead of the game if blocked.
//
// gameId/historyItems (optional): when provided, the words/questions the
// learner is about to play are recorded to that game's Firestore history
// the moment they actually start — not at generate time, so an abandoned
// generation never counts as "played". This is fire-and-forget; it never
// delays or blocks startFn().
async function gateAndStartGame(startFn, gameId, historyItems) {
  if (gameId && historyItems?.length) recordGameHistory(gameId, historyItems);
  startFn();
}

// ============================================================
// GROUP GAME — up to 6 players, each on their own phone, race to score the
// most correct answers within a shared 2-minute clock. A "room" (RTDB, see
// groupGame.js) syncs the lobby roster, which game/topic the host picked,
// when the clock started, and a live per-player score — every client
// generates/plays its OWN local copy of the underlying mini-game (Maze,
// Meltdown, Word Grid, or Connectors) using the exact same fetch/start
// functions those solo games already use, just reporting a point back to
// the room every time it registers a correct answer.
// ============================================================


groupGameChoiceCancelBtn.addEventListener('click', () => {
  groupGameChoiceModalOverlay.classList.remove('show');
});

groupGameHostOption.addEventListener('click', async () => {
  groupGameHostOption.classList.add('disabled');
  groupGameChoiceError.textContent = '';
  try {
    const u = auth.currentUser;
    if (!u) throw new Error('Not signed in.');
    const hostName = myGroupGamePlayerName();
    const code = await createRoom(u.uid, hostName);
    groupGameCode = code;
    groupGameRole = 'host';
    groupGameChoiceModalOverlay.classList.remove('show');
    openGroupGameLobby();
  } catch (err) {
    groupGameChoiceError.textContent = err.message || 'Could not create a group — try again.';
  } finally {
    groupGameHostOption.classList.remove('disabled');
  }
});

groupGameJoinOption.addEventListener('click', () => {
  groupGameChoiceModalOverlay.classList.remove('show');
  groupGameJoinError.textContent = '';
  groupGameJoinCodeInput.value = '';
  groupGameJoinModalOverlay.classList.add('show');
  setTimeout(() => groupGameJoinCodeInput.focus(), 50);
});
groupGameJoinCancelBtn.addEventListener('click', () => {
  groupGameJoinModalOverlay.classList.remove('show');
});
groupGameJoinCodeInput.addEventListener('input', () => {
  groupGameJoinCodeInput.value = groupGameJoinCodeInput.value.toUpperCase().replace(/[^A-Z0-9]/g, '');
});

async function submitGroupGameJoin(code) {
  const u = auth.currentUser;
  if (!u) { groupGameJoinError.textContent = 'Not signed in.'; return; }
  const { code: joinedCode } = await joinRoom(code, u.uid, myGroupGamePlayerName());
  groupGameCode = joinedCode;
  groupGameRole = 'guest';
  groupGameJoinModalOverlay.classList.remove('show');
  openGroupGameLobby();
}

groupGameJoinSubmitBtn.addEventListener('click', async () => {
  const code = groupGameJoinCodeInput.value.trim().toUpperCase();
  if (!code) { groupGameJoinError.textContent = 'Type a code first.'; return; }
  groupGameJoinError.textContent = '';
  groupGameJoinSubmitBtn.disabled = true;
  groupGameJoinSubmitBtn.textContent = 'Connecting…';
  try {
    await submitGroupGameJoin(code);
  } catch (err) {
    groupGameJoinError.textContent = err.message || "Couldn't join that group.";
  } finally {
    groupGameJoinSubmitBtn.disabled = false;
    groupGameJoinSubmitBtn.textContent = 'Join';
  }
});

groupGameJoinScanBtn.addEventListener('click', async () => {
  groupGameJoinError.textContent = '';
  groupGameJoinScanBtn.disabled = true;
  try {
    const scanned = await scanJoinCode({ statusText: "Point the camera at the group's QR code" });
    if (!scanned) return;
    await submitGroupGameJoin(scanned);
  } catch (err) {
    groupGameJoinError.textContent = err.message || 'Could not join that group.';
  } finally {
    groupGameJoinScanBtn.disabled = false;
  }
});

// ---- Lobby ----
function openGroupGameLobby() {
  groupGameResultsShown = false;
  groupGameLobbyStartBtn.style.display = groupGameRole === 'host' ? '' : 'none';
  groupGameLobbyWaitingNote.style.display = groupGameRole === 'host' ? 'none' : '';
  groupGameLobbySub.textContent = groupGameRole === 'host'
    ? 'Share this code with your friends'
    : "You're in! Waiting for the host…";
  groupGameLobbyCode.textContent = groupGameCode;
  renderJoinQr(groupGameLobbyQrCanvas, groupGameCode).catch(() => {});
  groupGameLobbyOverlay.classList.add('show');

  if (groupGameUnsubscribe) { groupGameUnsubscribe(); groupGameUnsubscribe = null; }
  groupGameUnsubscribe = listenToRoom(groupGameCode, onGroupRoomUpdate);
}

function renderGroupGameRoster(room) {
  const players = Object.entries(room.players || {});
  players.sort((a, b) => a[1].joinedAt - b[1].joinedAt);
  groupGameLobbyRoster.innerHTML = players.map(([uid, p]) => `
    <div class="group-roster-row">
      <span class="group-roster-dot" style="background:${p.color}"></span>
      <span class="group-roster-name">${escapeHtml(p.name)}</span>
      ${uid === room.hostUid ? '<span class="group-roster-host-tag">Host</span>' : ''}
    </div>
  `).join('');
  const remaining = 6 - players.length;
  groupGameLobbyEmptyNote.textContent = remaining > 0
    ? `Room for ${remaining} more player${remaining === 1 ? '' : 's'}.`
    : "Room's full!";
}

groupGameLobbyStartBtn.addEventListener('click', () => {
  groupGamePickerModalOverlay.classList.add('show');
});
groupGamePickerCancelBtn.addEventListener('click', () => {
  groupGamePickerModalOverlay.classList.remove('show');
});
groupGamePicker2MinOption.addEventListener('click', () => {
  groupGamePickerModalOverlay.classList.remove('show');
  openGroupGameSetupModal();
});

async function exitGroupGameEverywhere() {
  if (groupGameUnsubscribe) { groupGameUnsubscribe(); groupGameUnsubscribe = null; }
  if (groupGameCode && auth.currentUser) {
    leaveRoom(groupGameCode, auth.currentUser.uid).catch(() => {});
  }
  groupGameCode = null;
  groupGameRole = null;
  groupGameLatestRoom = null;
  groupGameActive = false;
  groupGameGameId = null;
  stopGroupGameCountdown();
  groupGameHud.classList.remove('show');
  [groupGameLobbyOverlay, groupGamePickerModalOverlay, groupGameSetupModalOverlay,
    groupGameRoundEndModalOverlay, groupGameResultsOverlay].forEach((el) => el.classList.remove('show'));
}

groupGameLobbyLeaveBtn.addEventListener('click', () => {
  if (groupGameRole === 'host') {
    cancelRoom(groupGameCode).catch(() => {});
  }
  exitGroupGameEverywhere();
});
groupGameResultsLeaveBtn.addEventListener('click', () => {
  exitGroupGameEverywhere();
});

// ---- "2 Minutes" setup (game + topic + difficulty) ----
function openGroupGameSetupModal() {
  groupGameSetupError.textContent = '';
  groupGamePendingGameId = 'maze';
  groupGameSetupCustomInput.value = '';
  groupGameSetupGameGrid.querySelectorAll('.group-setup-game-btn').forEach((btn) => {
    btn.classList.toggle('selected', btn.dataset.game === groupGamePendingGameId);
  });
  if (activeCourse) {
    groupGameSetupCourseOption.classList.remove('disabled');
    groupGameSetupCourseOptionDesc.textContent = activeCourse.description || '';
    selectGroupGameTopicOption('course');
  } else {
    groupGameSetupCourseOption.classList.add('disabled');
    selectGroupGameTopicOption('custom');
  }
  selectGroupGameDifficulty(groupGameDifficulty);
  groupGameSetupModalOverlay.classList.add('show');
}
groupGameSetupCancelBtn.addEventListener('click', () => {
  groupGameSetupModalOverlay.classList.remove('show');
});
groupGameSetupGameGrid.querySelectorAll('.group-setup-game-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    groupGamePendingGameId = btn.dataset.game;
    groupGameSetupGameGrid.querySelectorAll('.group-setup-game-btn').forEach((b) => {
      b.classList.toggle('selected', b === btn);
    });
  });
});
groupGameSetupCourseOption.addEventListener('click', () => {
  if (groupGameSetupCourseOption.classList.contains('disabled')) return;
  selectGroupGameTopicOption('course');
});
groupGameSetupCustomOption.addEventListener('click', () => selectGroupGameTopicOption('custom'));
groupGameSetupCustomInput.addEventListener('click', (e) => e.stopPropagation());
groupGameSetupCustomInput.addEventListener('input', () => selectGroupGameTopicOption('custom'));
function selectGroupGameTopicOption(which) {
  groupGameSetupCourseOption.classList.toggle('selected', which === 'course');
  groupGameSetupCustomOption.classList.toggle('selected', which === 'custom');
  if (which === 'custom') groupGameSetupCustomInput.focus();
}
groupGameSetupDifficultyRow.querySelectorAll('.maze-difficulty-btn').forEach((btn) => {
  btn.addEventListener('click', () => selectGroupGameDifficulty(btn.dataset.difficulty));
});
function selectGroupGameDifficulty(difficulty) {
  groupGameDifficulty = difficulty;
  groupGameSetupDifficultyRow.querySelectorAll('.maze-difficulty-btn').forEach((btn) => {
    btn.classList.toggle('selected', btn.dataset.difficulty === difficulty);
  });
}

groupGameSetupStartBtn.addEventListener('click', async () => {
  const isCustom = groupGameSetupCustomOption.classList.contains('selected');
  let topic;
  if (isCustom) {
    topic = groupGameSetupCustomInput.value.trim();
    if (!topic) { groupGameSetupError.textContent = 'Type a topic first.'; return; }
  } else {
    if (!activeCourse) { groupGameSetupError.textContent = 'Pick a course first.'; return; }
    topic = `${activeCourse.title}: ${activeCourse.description}`;
  }

  groupGameSetupError.textContent = '';
  groupGameSetupStartBtn.disabled = true;
  groupGameSetupStartBtn.textContent = 'Starting…';
  try {
    await startRound(groupGameCode, {
      gameId: groupGamePendingGameId, topic, difficulty: groupGameDifficulty, roundDurationMs: 120000,
    });
    groupGameSetupModalOverlay.classList.remove('show');
    // The room listener (onGroupRoomUpdate) picks up the 'playing' status
    // change and launches the local game for every client, host included.
  } catch (err) {
    groupGameSetupError.textContent = err.message || 'Could not start the round.';
  } finally {
    groupGameSetupStartBtn.disabled = false;
    groupGameSetupStartBtn.textContent = 'Start Round';
  }
});

// ---- Room listener — drives lobby roster, round launch, and results for
// every client (host and guests alike) off the single source of truth in
// RTDB. ----
function onGroupRoomUpdate(room) {
  groupGameLatestRoom = room;
  if (!room || room.status === 'ended') {
    if (groupGameCode) exitGroupGameEverywhere();
    return;
  }

  if (groupGameLobbyOverlay.classList.contains('show')) {
    renderGroupGameRoster(room);
  }

  if (room.status === 'playing' && !groupGameActive) {
    groupGameLobbyOverlay.classList.remove('show');
    groupGamePickerModalOverlay.classList.remove('show');
    groupGameSetupModalOverlay.classList.remove('show');
    groupGameResultsOverlay.classList.remove('show');
    launchGroupGameRound(room);
  }

  if (room.status === 'results') {
    showGroupGameResults(room);
  }

  if (groupGameHud.classList.contains('show')) {
    renderGroupGameHudPlayers(room);
  }
}

// ---- Launching + continuing the shared "2 Minutes" round ----
async function launchGroupGameRound(room) {
  groupGameActive = true;
  groupGameGameId = room.gameId;
  groupGameTopic = room.topic;
  groupGameDifficulty = room.difficulty || 'medium';
  groupGameRoundEndsAt = (room.roundStartedAt || Date.now()) + (room.roundDurationMs || 120000);
  groupGameResultsShown = false;
  startGroupGameCountdown();
  renderGroupGameHudPlayers(room);
  groupGameHud.classList.add('show');

  try {
    if (room.gameId === 'maze') await startTwoMinMaze();
    else if (room.gameId === 'meltdown') await startTwoMinMeltdown();
    else if (room.gameId === 'wordGrid') await startTwoMinWordGrid();
    else if (room.gameId === 'connectors') await startTwoMinConnectors();
  } catch (err) {
    // Generation failed — let the countdown carry on; the player just sits
    // out this round rather than the whole group being blocked.
    console.warn('Group Game: could not start local round', err);
  }
}

async function startTwoMinMaze() {
  const history = await getGameHistoryContext('maze');
  const data = await fetchMazeQuestions(groupGameTopic, history);
  mazeQuestions = data.questions;
  mazeChosenDifficulty = groupGameDifficulty;
  recordGameHistory('maze', mazeQuestions.map((q) => q.question));
  startMazeView();
}
async function startTwoMinMeltdown() {
  const history = await getGameHistoryContext('meltdown');
  const data = await fetchMeltdownQuestions(groupGameTopic, history);
  meltdownQuestions = data.questions;
  meltdownChosenDifficulty = groupGameDifficulty;
  recordGameHistory('meltdown', meltdownQuestions.map((q) => q.question));
  startMeltdownView();
}
async function startTwoMinWordGrid() {
  const history = await getGameHistoryContext('wordGrid');
  const data = await fetchWordGridWord(groupGameTopic, history);
  wordGridWord = data.word;
  wordGridHint = data.hint;
  wordGridTopic = groupGameTopic;
  recordGameHistory('wordGrid', [wordGridWord]);
  startWordGridView();
}
async function startTwoMinConnectors() {
  // Solo Connectors takes 1-2 topics (one per course/custom topic picked);
  // Group Game's setup only collects one topic (to match Maze/Meltdown/Word
  // Grid's simpler picker), so it's passed as a single-entry array — the
  // worker always builds 4 groups regardless, inventing extra ones itself
  // when fewer topics are given (see the contract note above
  // fetchConnectorsPuzzle).
  const topics = [{ label: groupGameTopic, description: '' }];
  const history = await getGameHistoryContext('connectors');
  const data = await fetchConnectorsPuzzle(topics, groupGameDifficulty, history);
  connectorsChosenDifficulty = groupGameDifficulty;
  setUpConnectorsGame(data.groups, topics);
  recordGameHistory('connectors', connectorsGroups.flatMap((g) => g.words));
  startConnectorsView();
}

// Called when an individual player's local mini-game instance finishes
// (maze solved, meltdown melted, word guessed/missed, board solved/lost)
// while the shared clock is still running — shows a quick "Keep Going"
// prompt instead of that game's normal solo end screen.
function showGroupGameRoundEnd(subtitle) {
  if (Date.now() >= groupGameRoundEndsAt) {
    // Clock already ran out — go straight to results instead of a
    // Keep Going prompt that would have nowhere to go.
    finishGroupGameLocally();
    return;
  }
  groupGameRoundEndSub.textContent = subtitle || "The clock's still running — keep going!";
  groupGameRoundEndModalOverlay.classList.add('show');
}

groupGameRoundEndBtn.addEventListener('click', async () => {
  groupGameRoundEndModalOverlay.classList.remove('show');
  const gameId = groupGameGameId;
  try {
    if (gameId === 'maze') startMazeView();
    else if (gameId === 'meltdown') startMeltdownView();
    else if (gameId === 'wordGrid') await startTwoMinWordGrid();
    else if (gameId === 'connectors') await startTwoMinConnectors();
  } catch (err) {
    console.warn('Group Game: could not continue round', err);
  }
});

// ---- Shared countdown, HUD, and results ----
function startGroupGameCountdown() {
  stopGroupGameCountdown();
  updateGroupGameHudTimer();
  groupGameCountdownHandle = setInterval(updateGroupGameHudTimer, 250);
}
function stopGroupGameCountdown() {
  if (groupGameCountdownHandle) { clearInterval(groupGameCountdownHandle); groupGameCountdownHandle = null; }
}
function updateGroupGameHudTimer() {
  const remainingMs = Math.max(0, groupGameRoundEndsAt - Date.now());
  const totalSeconds = Math.ceil(remainingMs / 1000);
  const mm = Math.floor(totalSeconds / 60);
  const ss = totalSeconds % 60;
  groupGameHudTimer.textContent = `${mm}:${String(ss).padStart(2, '0')}`;
  groupGameHudTimer.classList.toggle('urgent', totalSeconds <= 15);
  if (remainingMs <= 0) {
    stopGroupGameCountdown();
    finishGroupGameLocally();
  }
}
function renderGroupGameHudPlayers(room) {
  const players = Object.entries(room.players || {});
  players.sort((a, b) => (b[1].score || 0) - (a[1].score || 0));
  groupGameHudPlayers.innerHTML = players.map(([, p]) => `
    <div class="group-hud-player">
      <span class="group-hud-player-dot" style="background:${p.color}"></span>
      <span class="group-hud-player-score">${p.score || 0}</span>
    </div>
  `).join('');
}

// Called locally once this device's countdown hits zero — stops whatever
// mini-game is currently open and (host only) flips the room to
// 'results' so every device converges there via the listener.
function finishGroupGameLocally() {
  if (!groupGameActive) return;
  groupGameActive = false;
  groupGameHud.classList.remove('show');
  groupGameRoundEndModalOverlay.classList.remove('show');
  // Close whichever mini-game view might still be open, same cleanup each
  // game's own Exit button does.
  mazeActive = false; mazeAwaitingAnswer = false; clearInterval(mazeTimerHandle);
  document.removeEventListener('keydown', handleMazeKeydown);
  mazeViewOverlay.classList.remove('show');
  mazeQuestionModalOverlay.classList.remove('show');
  meltdownActive = false; clearInterval(meltdownTickHandle); clearTimeout(meltdownTimeoutHandle);
  meltdownViewOverlay.classList.remove('show');
  wordGridActive = false; wordGridViewOverlay.classList.remove('show');
  connectorsViewOverlay.classList.remove('show');

  if (groupGameRole === 'host' && groupGameCode) {
    endRound(groupGameCode).catch(() => {});
  }
  if (groupGameLatestRoom) showGroupGameResults(groupGameLatestRoom);
}

function showGroupGameResults(room) {
  if (groupGameResultsShown) return;
  groupGameResultsShown = true;
  stopGroupGameCountdown();
  groupGameHud.classList.remove('show');
  groupGameRoundEndModalOverlay.classList.remove('show');

  const players = Object.entries(room.players || {});
  players.sort((a, b) => (b[1].score || 0) - (a[1].score || 0));
  const topScore = players.length ? (players[0][1].score || 0) : 0;
  groupGameResultsList.innerHTML = players.map(([, p], i) => `
    <div class="group-results-row${p.score === topScore && topScore > 0 ? ' winner' : ''}">
      <span class="group-results-rank">${i + 1}</span>
      <span class="group-results-dot" style="background:${p.color}"></span>
      <span class="group-results-name">${escapeHtml(p.name)}</span>
      <span class="group-results-score">${p.score || 0}</span>
    </div>
  `).join('');

  groupGamePlayAgainBtn.style.display = groupGameRole === 'host' ? '' : 'none';
  groupGameResultsWaitingNote.style.display = groupGameRole === 'host' ? 'none' : '';
  groupGameResultsOverlay.classList.add('show');
}

groupGamePlayAgainBtn.addEventListener('click', async () => {
  try {
    await resetRoomForNewRound(groupGameCode);
    groupGameResultsOverlay.classList.remove('show');
    groupGameResultsShown = false;
    groupGamePickerModalOverlay.classList.add('show');
  } catch (err) {
    console.warn('Group Game: could not reset for a new round', err);
  }
});

function openTriviaChooseModal() {
  triviaChooseError.textContent = '';
  triviaCustomInput.value = '';
  triviaReady = false;
  triviaQuestions = [];
  triviaGenerateBtn.disabled = false;
  triviaGenerateBtn.textContent = 'Generate';
  triviaChooseTitle.textContent = triviaVoiceMode ? 'Choose Your Voice Trivia' : 'Choose Your Trivia';

  if (activeCourse) {
    triviaCourseOption.classList.remove('disabled');
    triviaCourseOptionDesc.textContent = activeCourse.description || '';
    selectTriviaOption('course');
  } else {
    triviaCourseOption.classList.add('disabled');
    selectTriviaOption('custom');
  }
  triviaChooseModalOverlay.classList.add('show');
}

triviaChooseCancelBtn.addEventListener('click', () => {
  triviaChooseModalOverlay.classList.remove('show');
});

triviaCourseOption.addEventListener('click', () => {
  if (triviaCourseOption.classList.contains('disabled')) return;
  selectTriviaOption('course');
});
triviaCustomOption.addEventListener('click', () => selectTriviaOption('custom'));
triviaCustomInput.addEventListener('click', (e) => e.stopPropagation()); // avoid double-toggling via bubbled click
triviaCustomInput.addEventListener('input', () => {
  selectTriviaOption('custom');
  resetTriviaReadyState();
});

function selectTriviaOption(which) {
  triviaCourseOption.classList.toggle('selected', which === 'course');
  triviaCustomOption.classList.toggle('selected', which === 'custom');
  if (which === 'custom') triviaCustomInput.focus();
  resetTriviaReadyState();
}

// If the learner changes their selection after already generating a set,
// that set no longer matches — fall back to needing a fresh Generate press.
function resetTriviaReadyState() {
  if (!triviaReady) return;
  triviaReady = false;
  triviaQuestions = [];
  triviaGenerateBtn.disabled = false;
  triviaGenerateBtn.textContent = 'Generate';
}

triviaGenerateBtn.addEventListener('click', async () => {
  if (triviaReady) {
    triviaChooseModalOverlay.classList.remove('show');
    const startFn = triviaVoiceMode ? startVoiceTriviaView : startTriviaView;
    gateAndStartGame(startFn, 'trivia', triviaQuestions.map((q) => q.question));
    return;
  }

  const isCustom = triviaCustomOption.classList.contains('selected');
  let topic;
  if (isCustom) {
    topic = triviaCustomInput.value.trim();
    if (!topic) {
      triviaChooseError.textContent = 'Type a topic first.';
      return;
    }
  } else {
    if (!activeCourse) {
      triviaChooseError.textContent = 'Pick a course first.';
      return;
    }
    topic = `${activeCourse.title}: ${activeCourse.description}`;
  }

  triviaChooseError.textContent = '';
  triviaGenerateBtn.disabled = true;
  triviaGenerateBtn.textContent = 'Generating your trivia…';

  try {
    const history = await getGameHistoryContext('trivia');
    const data = await fetchTrivia(topic, history);
    triviaQuestions = data.questions;
    triviaLastTopic = topic;
    triviaReady = true;
    triviaGenerateBtn.disabled = false;
    triviaGenerateBtn.textContent = 'Start';
  } catch (err) {
    triviaChooseError.textContent = err.message || 'Something went wrong. Try again.';
    triviaGenerateBtn.disabled = false;
    triviaGenerateBtn.textContent = 'Generate';
  }
});

// Calls the worker for a 20-question trivia set on the given topic.
// `history` — [{ text, daysAgo }, ...] of questions this learner has played
// before across past sessions (see gameHistory.js) — tells the worker's AI
// prompt to avoid recently-played questions and only reuse older ones if it
// runs out of fresh material.
async function fetchTrivia(topic, history) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'generateTrivia', topic, history }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Could not create trivia.');
  if (!Array.isArray(data.questions) || data.questions.length !== 20) {
    throw new Error('Trivia generation failed — try again.');
  }
  return data;
}

// ---- Full-screen trivia player (TikTok/Shorts-style vertical scroll-snap) ----
function startTriviaView() {
  const color = activeCourse ? activeCourse.color : '#1E6FE0';
  triviaViewContainer.style.setProperty('--cc', color);
  renderTriviaSlides();
  triviaViewOverlay.classList.add('show');
  triviaScroller.scrollTop = 0;
}

function renderTriviaSlides() {
  const questionSlides = triviaQuestions.map((q, i) => `
    <div class="trivia-slide" data-index="${i}">
      <div class="trivia-question">${escapeHtml(q.question)}</div>
      <button class="trivia-answer-btn" data-index="${i}">Show Answer</button>
      <div class="trivia-answer-text" style="display:none"></div>
      ${i === 0 ? `
        <div class="trivia-scroll-hint">
          <span class="material-symbols-outlined">keyboard_arrow_up</span>
          Scroll up for next
        </div>` : ''}
    </div>
  `).join('');

  const finishedSlide = `
    <div class="trivia-slide trivia-finished-slide">
      <div class="trivia-finished-title">You Finished! 🎉</div>
      <div class="trivia-finished-actions">
        <button class="trivia-finished-btn-primary" id="triviaGenerateMoreBtn">Generate More</button>
        <button class="trivia-finished-btn-secondary" id="triviaExitFinishedBtn">Exit</button>
      </div>
      <div class="trivia-finished-error" id="triviaFinishedError"></div>
    </div>
  `;

  triviaScroller.innerHTML = questionSlides + finishedSlide;

  triviaScroller.querySelectorAll('.trivia-answer-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const slide = btn.closest('.trivia-slide');
      const answerEl = slide.querySelector('.trivia-answer-text');
      const q = triviaQuestions[Number(btn.dataset.index)];
      answerEl.textContent = q.answer;
      answerEl.style.display = 'block';
      btn.style.display = 'none';
    });
  });

  triviaScroller.querySelector('#triviaGenerateMoreBtn')?.addEventListener('click', () => { checkGameBadge('trivia'); generateMoreTrivia(); });
  triviaScroller.querySelector('#triviaExitFinishedBtn')?.addEventListener('click', () => { checkGameBadge('trivia'); closeTriviaView(); });
}

async function generateMoreTrivia() {
  const btn = triviaScroller.querySelector('#triviaGenerateMoreBtn');
  const secondaryBtn = triviaScroller.querySelector('#triviaExitFinishedBtn');
  const errorEl = triviaScroller.querySelector('#triviaFinishedError');
  if (!btn || !triviaLastTopic) return;

  btn.disabled = true;
  if (secondaryBtn) secondaryBtn.disabled = true;
  btn.textContent = 'Generating more…';
  if (errorEl) errorEl.textContent = '';

  try {
    const history = await getGameHistoryContext('trivia');
    const data = await fetchTrivia(triviaLastTopic, history);
    const startIndex = triviaQuestions.length;
    triviaQuestions = triviaQuestions.concat(data.questions);
    renderTriviaSlides();
    const target = triviaScroller.querySelector(`.trivia-slide[data-index="${startIndex}"]`);
    if (target) target.scrollIntoView({ behavior: 'auto', block: 'start' });
  } catch (err) {
    btn.disabled = false;
    if (secondaryBtn) secondaryBtn.disabled = false;
    btn.textContent = 'Generate More';
    if (errorEl) errorEl.textContent = err.message || 'Could not generate more trivia.';
  }
}

function closeTriviaView() {
  triviaViewOverlay.classList.remove('show');
  triviaQuestions = [];
  triviaReady = false;
  triviaLastTopic = null;
}

triviaExitBtn.addEventListener('click', closeTriviaView);

// ============================================================
// VOICE TRIVIA — same question sets as Trivia (see fetchTrivia above /
// triviaChooseModal), but played as a voice-first game: the question is
// read aloud (TTS), the app listens on the mic (STT) and re-checks the
// spoken answer against the AI grader every ~5s (see gradeVoiceAnswer in
// the Learn Worker), until it's correct or the learner gives up.
// ============================================================
function wait(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

// Speaks text aloud with the browser/WebView's built-in speech synthesis —
// free, on-device, no worker call. Resolves once speech finishes (or
// immediately if speech synthesis isn't available at all).
function vtSpeak(text) {
  return new Promise((resolve) => {
    if (!('speechSynthesis' in window) || !text) { resolve(); return; }
    window.speechSynthesis.cancel(); // don't let overlapping utterances queue up
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 0.95;
    utter.onend = () => resolve();
    utter.onerror = () => resolve();
    window.speechSynthesis.speak(utter);
  });
}

// Two-tone WebAudio chime — no audio asset needed, so "free" the same way
// the TTS/STT are. success=true is a short rising chime, false is a low buzz.
function vtPlayChime(success) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = success ? 880 : 220;
    gain.gain.value = 0.16;
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.frequency.linearRampToValueAtTime(success ? 1320 : 180, ctx.currentTime + 0.18);
    gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.32);
    osc.stop(ctx.currentTime + 0.34);
    osc.onended = () => ctx.close();
  } catch { /* WebAudio unavailable — silently skip the chime, not critical */ }
}

function vtUpdateMicUI(listening, statusText) {
  voiceTriviaMic.classList.toggle('listening', listening);
  if (statusText !== undefined) voiceTriviaStatus.textContent = statusText;
}

// Picks the best audio format MediaRecorder actually supports on this
// device — Chrome/Android typically support webm/opus, Safari/WKWebView
// typically support mp4/aac instead, so this can't be hardcoded.
function vtPickMimeType() {
  const candidates = [
    { mime: 'audio/webm;codecs=opus', ext: 'webm' },
    { mime: 'audio/webm', ext: 'webm' },
    { mime: 'audio/mp4', ext: 'mp4' },
    { mime: 'audio/aac', ext: 'aac' },
  ];
  for (const c of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(c.mime)) return c;
  }
  return { mime: '', ext: 'webm' }; // let the browser pick whatever its default is
}

function vtBlobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve((reader.result || '').toString().split(',')[1] || '');
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function vtPostAudio(action, extraFields, base64, mimeType) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, ...extraFields, audio: base64, mimeType }),
  });
  return res.json();
}

// Requests mic access once and keeps the stream open across the whole
// session (re-prompting for permission on every question would be jarring).
// Plain getUserMedia — no native plugin, works the same in a Capacitor
// WKWebView, desktop Chrome, or mobile Safari.
async function vtEnsureMic() {
  if (vtMicStream) return true;
  try {
    vtMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    vtMimeInfo = vtPickMimeType();
    return true;
  } catch (err) {
    console.warn('Mic permission denied/unavailable:', err);
    return false;
  }
}

function vtReleaseMic() {
  if (vtMicStream) {
    vtMicStream.getTracks().forEach((t) => t.stop());
    vtMicStream = null;
  }
}

// Records one ~5s chunk, posts it to `workerAction` (plus extraFields) for
// transcription (and, for the question loop, grading in the same call),
// hands the parsed result to onResult, and — unless onResult returns
// `false` or the loop/session has been stopped in the meantime — records
// another chunk immediately after. This single loop drives both the live
// question-answering ("transcribeAndGrade") and the end-screen voice
// commands ("transcribeAudio" + local keyword match).
async function vtRecordChunkLoop(mySession, workerAction, extraFields, onResult) {
  if (mySession !== vtSessionId || !vtRecordingLoopActive) return;
  if (!vtMicStream) {
    vtUpdateMicUI(false, "Couldn't access the microphone — tap below to give up.");
    return;
  }

  const chunks = [];
  let recorder;
  try {
    recorder = new MediaRecorder(vtMicStream, vtMimeInfo?.mime ? { mimeType: vtMimeInfo.mime } : undefined);
  } catch (err) {
    console.warn('MediaRecorder failed to start:', err);
    vtUpdateMicUI(false, "Couldn't start recording — tap below to give up.");
    return;
  }
  vtMediaRecorder = recorder;
  recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunks.push(e.data); };

  recorder.onstop = async () => {
    if (mySession !== vtSessionId || !vtRecordingLoopActive) return;
    if (!chunks.length) { vtRecordChunkLoop(mySession, workerAction, extraFields, onResult); return; }

    vtUpdateMicUI(true, 'Thinking…');
    try {
      const blob = new Blob(chunks, { type: vtMimeInfo?.mime || 'audio/webm' });
      const base64 = await vtBlobToBase64(blob);
      const data = await vtPostAudio(workerAction, extraFields, base64, vtMimeInfo?.mime || 'audio/webm');
      if (mySession !== vtSessionId || !vtRecordingLoopActive) return;
      const keepGoing = await onResult(data);
      if (keepGoing === false) return;
    } catch (err) {
      console.warn('Voice audio round failed:', err);
    }
    if (mySession === vtSessionId && vtRecordingLoopActive) {
      vtRecordChunkLoop(mySession, workerAction, extraFields, onResult);
    }
  };

  vtUpdateMicUI(true, 'Listening…');
  try {
    recorder.start();
  } catch (err) {
    vtUpdateMicUI(false, "Couldn't start recording — tap below to give up.");
    return;
  }
  setTimeout(() => { if (recorder.state === 'recording') recorder.stop(); }, 5000);
}

function vtStopListening() {
  vtRecordingLoopActive = false;
  vtUpdateMicUI(false, '');
  if (vtMediaRecorder && vtMediaRecorder.state === 'recording') {
    vtMediaRecorder.onstop = null; // don't let a trailing stop-event kick off another chunk
    try { vtMediaRecorder.stop(); } catch { /* already stopped */ }
  }
  vtMediaRecorder = null;
}

// Starts the record→transcribe→grade loop for the current question.
async function vtStartQuestionListening(mySession) {
  const ok = await vtEnsureMic();
  if (mySession !== vtSessionId) return;
  if (!ok) {
    vtUpdateMicUI(false, "Couldn't access the microphone — tap below to give up.");
    return;
  }
  vtRecordingLoopActive = true;
  vtRecordChunkLoop(mySession, 'transcribeAndGrade', {
    question: triviaQuestions[vtIndex].question,
    correctAnswer: triviaQuestions[vtIndex].answer,
  }, async (data) => {
    if (data.transcript) voiceTriviaTranscript.textContent = data.transcript;
    if (data.correct) { await handleVoiceCorrect(mySession); return false; }
    if (data.gaveUp) { await handleVoiceGiveUp(mySession); return false; }
    return true;
  });
}

function startVoiceTriviaView() {
  vtSessionId++;
  const color = activeCourse ? activeCourse.color : '#1E6FE0';
  voiceTriviaContainer.style.setProperty('--cc', color);
  voiceTriviaFinished.style.display = 'none';
  voiceTriviaPlayArea.style.display = 'block';
  voiceTriviaViewOverlay.classList.add('show');
  playVoiceTriviaQuestion(0);
}

async function playVoiceTriviaQuestion(i) {
  const mySession = vtSessionId;
  if (i >= triviaQuestions.length) { finishVoiceTrivia(); return; }
  vtIndex = i;
  const q = triviaQuestions[i];

  voiceTriviaProgress.textContent = `Question ${i + 1} of ${triviaQuestions.length}`;
  voiceTriviaQuestion.textContent = q.question;
  voiceTriviaQuestion.classList.remove('flash-correct', 'flash-giveup');
  voiceTriviaTranscript.textContent = '';
  voiceTriviaFeedback.textContent = '';
  vtUpdateMicUI(false, 'Reading question…');

  await vtSpeak(q.question);
  if (mySession !== vtSessionId) return; // exited (or restarted) mid-speech

  vtStartQuestionListening(mySession);
}

async function handleVoiceCorrect(mySession) {
  vtStopListening();
  voiceTriviaQuestion.classList.add('flash-correct');
  voiceTriviaFeedback.textContent = "That's right! ✅";
  vtPlayChime(true);
  await vtSpeak("That's right!");
  if (mySession !== vtSessionId) return;
  await wait(500);
  if (mySession !== vtSessionId) return;
  playVoiceTriviaQuestion(vtIndex + 1);
}

async function handleVoiceGiveUp(mySession) {
  vtStopListening();
  voiceTriviaQuestion.classList.add('flash-giveup');
  const answer = triviaQuestions[vtIndex].answer;
  voiceTriviaFeedback.textContent = `The answer was: ${answer}`;
  vtPlayChime(false);
  await vtSpeak(`The answer was ${answer}.`);
  if (mySession !== vtSessionId) return;
  await wait(400);
  if (mySession !== vtSessionId) return;
  playVoiceTriviaQuestion(vtIndex + 1);
}

// Tap fallback for "give up" — same handler the AI grader's gaveUp:true
// path uses, just triggered manually instead of by voice.
voiceTriviaGiveUpBtn.addEventListener('click', () => {
  if (!voiceTriviaViewOverlay.classList.contains('show') || voiceTriviaFinished.style.display !== 'none') return;
  handleVoiceGiveUp(vtSessionId);
});

async function finishVoiceTrivia() {
  const mySession = vtSessionId;
  vtStopListening();
  voiceTriviaPlayArea.style.display = 'none';
  voiceTriviaFinished.style.display = 'flex';
  await checkGameBadge('voiceTrivia');

  await vtSpeak('You finished! Say play again, or exit.');
  if (mySession !== vtSessionId) return;

  // Voice controls at the end screen — listens for "play again" / "exit"
  // as a spoken alternative to tapping the buttons. Simple local keyword
  // matching is enough here (low stakes, plain transcription, no AI grading
  // needed — that's why this uses the 'transcribeAudio' action, not
  // 'transcribeAndGrade').
  const ok = await vtEnsureMic();
  if (mySession !== vtSessionId || !ok) return;
  vtRecordingLoopActive = true;
  vtRecordChunkLoop(mySession, 'transcribeAudio', {}, async (data) => {
    const t = (data.transcript || '').toLowerCase();
    if (t.includes('play again') || t.includes('again')) {
      vtStopListening();
      voiceTriviaPlayAgain();
      return false;
    }
    if (t.includes('exit') || t.includes('quit') || t.includes('stop') || t.includes('done')) {
      vtStopListening();
      closeVoiceTriviaView();
      return false;
    }
    return true;
  });
}

async function voiceTriviaPlayAgain() {
  const mySession = vtSessionId;
  voiceTriviaFinished.style.display = 'none';
  voiceTriviaPlayArea.style.display = 'block';
  voiceTriviaQuestion.textContent = '';
  voiceTriviaFeedback.textContent = '';
  vtUpdateMicUI(false, 'Loading new questions…');
  try {
    if (triviaLastTopic) {
      const history = await getGameHistoryContext('trivia');
      const data = await fetchTrivia(triviaLastTopic, history);
      if (mySession !== vtSessionId) return;
      triviaQuestions = data.questions;
      recordGameHistory('trivia', triviaQuestions.map((q) => q.question));
    }
    playVoiceTriviaQuestion(0);
  } catch (err) {
    vtUpdateMicUI(false, err.message || 'Could not load new questions.');
  }
}

function closeVoiceTriviaView() {
  vtSessionId++; // invalidates any in-flight async work tied to the round that just ended
  vtStopListening();
  vtReleaseMic();
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  voiceTriviaViewOverlay.classList.remove('show');
  triviaQuestions = [];
  triviaReady = false;
  triviaLastTopic = null;
  triviaVoiceMode = false;
}

voiceTriviaExitBtn.addEventListener('click', closeVoiceTriviaView);
voiceTriviaExitFinishedBtn.addEventListener('click', closeVoiceTriviaView);
voiceTriviaPlayAgainBtn.addEventListener('click', () => {
  vtStopListening();
  voiceTriviaPlayAgain();
});

// ============================================================
// "WHY?" EXPLANATION MODAL — explains a missed question on demand
// ============================================================
lessonWhyBtn.addEventListener('click', async () => {
  if (!missedQuestion) return;
  if (!isPremium()) {
    openPaywall({ reason: 'Explain My Answer is a Premium feature.' });
    return;
  }
  whyModalBody.textContent = 'Thinking…';
  whyModalOverlay.classList.add('show');

  try {
    const res = await fetch(LEARN_WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'explainAnswer',
        courseTitle: activeCourse?.title || '',
        lessonTitle: currentLesson?.lessonTitle || '',
        question: missedQuestion.question,
        choices: missedQuestion.choices,
        correctIndex: missedQuestion.correctIndex,
        selectedIndex: missedQuestion.selectedIndex,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Could not load an explanation.');
    whyModalBody.textContent = data.explanation;
  } catch (err) {
    whyModalBody.textContent = err.message || 'Could not load an explanation — try again.';
  }
});

whyModalCloseBtn.addEventListener('click', () => whyModalOverlay.classList.remove('show'));

// ============================================================
// AI ASSISTANT — small in-lesson chatbot, tucked in the bottom-right corner
// ============================================================
aiAssistantFab.addEventListener('click', () => {
  if (!isPremium()) {
    openPaywall({ reason: 'The AI Assistant is a Premium feature.' });
    return;
  }
  aiAssistantPanel.classList.add('show');
  aiAssistantFab.style.display = 'none';
  aiAssistantInput.focus();
});

aiAssistantCloseBtn.addEventListener('click', closeAiAssistantPanel);

function closeAiAssistantPanel() {
  aiAssistantPanel.classList.remove('show');
  aiAssistantFab.style.display = '';
}

// Wipes the chat thread — called whenever a lesson starts/exits so one
// lesson's conversation doesn't bleed into the next.
function resetAiAssistant() {
  aiAssistantHistory = [];
  aiAssistantBusy = false;
  aiAssistantMessages.innerHTML = '';
  aiAssistantInput.value = '';
  aiAssistantPanel.classList.remove('show');
  aiAssistantFab.style.display = '';
}

function renderAiAssistantMessages() {
  aiAssistantMessages.innerHTML = aiAssistantHistory.map((m) => `
    <div class="ai-assistant-msg ${m.role}">${escapeHtml(m.content)}</div>
  `).join('');
  aiAssistantMessages.scrollTop = aiAssistantMessages.scrollHeight;
}

aiAssistantForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (aiAssistantBusy) return;

  const question = aiAssistantInput.value.trim();
  if (!question) return;

  aiAssistantInput.value = '';
  aiAssistantHistory.push({ role: 'user', content: question });
  renderAiAssistantMessages();

  aiAssistantBusy = true;
  aiAssistantSendBtn.disabled = true;
  aiAssistantHistory.push({ role: 'assistant', content: '…', pending: true });
  renderAiAssistantMessages();

  try {
    const res = await fetch(LEARN_WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'chatAsk',
        courseTitle: activeCourse?.title || '',
        courseDescription: activeCourse?.description || '',
        lessonTitle: currentLesson?.lessonTitle || '',
        question,
        history: aiAssistantHistory.filter((m) => !m.pending),
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || "Couldn't get a reply.");

    aiAssistantHistory = aiAssistantHistory.filter((m) => !m.pending);
    aiAssistantHistory.push({ role: 'assistant', content: data.reply });
  } catch (err) {
    aiAssistantHistory = aiAssistantHistory.filter((m) => !m.pending);
    aiAssistantHistory.push({ role: 'assistant', content: err.message || "Sorry, I couldn't answer that." });
  } finally {
    aiAssistantBusy = false;
    aiAssistantSendBtn.disabled = false;
    renderAiAssistantMessages();
  }
});

// ============================================================
// MAZE — navigate a generated maze; every X seconds (by difficulty),
// answer a question to keep moving.
// ============================================================

// ---- Choose modal (topic + difficulty), mirrors the trivia choose flow ----
function openMazeChooseModal() {
  mazeChooseError.textContent = '';
  mazeCustomInput.value = '';
  mazeReady = false;
  mazeQuestions = [];
  mazeGenerateBtn.disabled = false;
  mazeGenerateBtn.textContent = 'Generate';

  if (activeCourse) {
    mazeCourseOption.classList.remove('disabled');
    mazeCourseOptionDesc.textContent = activeCourse.description || '';
    selectMazeTopicOption('course');
  } else {
    mazeCourseOption.classList.add('disabled');
    selectMazeTopicOption('custom');
  }
  selectMazeDifficulty(mazeChosenDifficulty);
  mazeChooseModalOverlay.classList.add('show');
}

mazeChooseCancelBtn.addEventListener('click', () => {
  mazeChooseModalOverlay.classList.remove('show');
});

mazeCourseOption.addEventListener('click', () => {
  if (mazeCourseOption.classList.contains('disabled')) return;
  selectMazeTopicOption('course');
});
mazeCustomOption.addEventListener('click', () => selectMazeTopicOption('custom'));
mazeCustomInput.addEventListener('click', (e) => e.stopPropagation());
mazeCustomInput.addEventListener('input', () => {
  selectMazeTopicOption('custom');
  resetMazeReadyState();
});

function selectMazeTopicOption(which) {
  mazeCourseOption.classList.toggle('selected', which === 'course');
  mazeCustomOption.classList.toggle('selected', which === 'custom');
  if (which === 'custom') mazeCustomInput.focus();
  resetMazeReadyState();
}

mazeDifficultyBtns.forEach((btn) => {
  btn.addEventListener('click', () => selectMazeDifficulty(btn.dataset.difficulty));
});

function selectMazeDifficulty(difficulty) {
  mazeChosenDifficulty = difficulty;
  mazeDifficultyBtns.forEach((btn) => btn.classList.toggle('selected', btn.dataset.difficulty === difficulty));
  resetMazeReadyState();
}

// A changed selection after generating a set means that set no longer
// matches — fall back to needing a fresh Generate press.
function resetMazeReadyState() {
  if (!mazeReady) return;
  mazeReady = false;
  mazeQuestions = [];
  mazeGenerateBtn.disabled = false;
  mazeGenerateBtn.textContent = 'Generate';
}

mazeGenerateBtn.addEventListener('click', async () => {
  if (mazeReady) {
    mazeChooseModalOverlay.classList.remove('show');
    gateAndStartGame(startMazeView, 'maze', mazeQuestions.map((q) => q.question));
    return;
  }

  const isCustom = mazeCustomOption.classList.contains('selected');
  let topic;
  if (isCustom) {
    topic = mazeCustomInput.value.trim();
    if (!topic) {
      mazeChooseError.textContent = 'Type a topic first.';
      return;
    }
  } else {
    if (!activeCourse) {
      mazeChooseError.textContent = 'Pick a course first.';
      return;
    }
    topic = `${activeCourse.title}: ${activeCourse.description}`;
  }

  mazeChooseError.textContent = '';
  mazeGenerateBtn.disabled = true;
  mazeGenerateBtn.textContent = 'Generating your maze…';

  try {
    const history = await getGameHistoryContext('maze');
    const data = await fetchMazeQuestions(topic, history);
    mazeQuestions = data.questions;
    mazeReady = true;
    mazeGenerateBtn.disabled = false;
    mazeGenerateBtn.textContent = 'Start Maze';
    mazeChooseModalOverlay.classList.remove('show');
    gateAndStartGame(startMazeView, 'maze', mazeQuestions.map((q) => q.question));
  } catch (err) {
    mazeGenerateBtn.disabled = false;
    mazeGenerateBtn.textContent = 'Generate';
    mazeChooseError.textContent = err.message || 'Could not generate maze questions.';
  }
});

// `history` — [{ text, daysAgo }, ...] previously-played maze questions (see
// gameHistory.js) so the worker's AI prompt can avoid recent repeats.
async function fetchMazeQuestions(topic, history) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'generateMazeQuestions', topic, history }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Could not generate maze questions.');
  return data;
}

// ---- Maze generation: recursive-backtracker carves a perfect maze (one
// unique path between any two cells) into an all-walls grid ----
function generateMazeGrid(size) {
  const cells = [];
  for (let r = 0; r < size; r++) {
    const row = [];
    for (let c = 0; c < size; c++) {
      row.push({ r, c, visited: false, walls: { top: true, right: true, bottom: true, left: true } });
    }
    cells.push(row);
  }

  const DIRS = [
    { name: 'top', dr: -1, dc: 0, opposite: 'bottom' },
    { name: 'right', dr: 0, dc: 1, opposite: 'left' },
    { name: 'bottom', dr: 1, dc: 0, opposite: 'top' },
    { name: 'left', dr: 0, dc: -1, opposite: 'right' },
  ];

  const stack = [cells[0][0]];
  cells[0][0].visited = true;

  while (stack.length) {
    const current = stack[stack.length - 1];
    const neighbors = [];
    for (const d of DIRS) {
      const nr = current.r + d.dr;
      const nc = current.c + d.dc;
      if (nr >= 0 && nr < size && nc >= 0 && nc < size && !cells[nr][nc].visited) {
        neighbors.push({ cell: cells[nr][nc], dir: d });
      }
    }
    if (neighbors.length) {
      const { cell: next, dir } = neighbors[Math.floor(Math.random() * neighbors.length)];
      current.walls[dir.name] = false;
      next.walls[dir.opposite] = false;
      next.visited = true;
      stack.push(next);
    } else {
      stack.pop();
    }
  }

  return cells;
}

// Fisher-Yates, used to shuffle the question set once per maze so the order
// isn't predictable across attempts.
function shuffleArray(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function startMazeView() {
  const { size, interval } = MAZE_DIFFICULTY[mazeChosenDifficulty];
  mazeSize = size;
  mazeIntervalSeconds = interval;
  mazeGridCells = generateMazeGrid(size);
  mazePath = [{ r: 0, c: 0 }];
  mazeQuestions = shuffleArray(mazeQuestions);
  mazeQuestionIndex = 0;
  mazeAwaitingAnswer = false;
  mazeActive = true;
  mazeXp = 120;
  mazeStartTime = Date.now();

  mazeViewTitle.textContent = `Maze — ${mazeChosenDifficulty[0].toUpperCase()}${mazeChosenDifficulty.slice(1)}`;
  renderMazeGrid();
  mazeViewOverlay.classList.add('show');
  document.addEventListener('keydown', handleMazeKeydown);
  startMazeTimer();
}

function renderMazeGrid() {
  mazeGrid.style.gridTemplateColumns = `repeat(${mazeSize}, 1fr)`;
  mazeGrid.style.gridTemplateRows = `repeat(${mazeSize}, 1fr)`;

  const player = mazePath[mazePath.length - 1];
  const goal = { r: mazeSize - 1, c: mazeSize - 1 };

  let html = '';
  for (let r = 0; r < mazeSize; r++) {
    for (let c = 0; c < mazeSize; c++) {
      const cell = mazeGridCells[r][c];
      const isPlayer = player.r === r && player.c === c;
      const isGoal = goal.r === r && goal.c === c;
      const borderStyle = (open) => open ? 'none' : '2px solid var(--ink)';
      const style = [
        `border-top:${borderStyle(!cell.walls.top)}`,
        `border-right:${borderStyle(!cell.walls.right)}`,
        `border-bottom:${borderStyle(!cell.walls.bottom)}`,
        `border-left:${borderStyle(!cell.walls.left)}`,
      ].join(';');
      html += `<div class="maze-cell${isGoal ? ' is-goal' : ''}" style="${style}">`;
      if (isGoal) html += `<span class="maze-cell-flag material-symbols-outlined">flag</span>`;
      if (isPlayer) html += `<div class="maze-cell-dot"></div>`;
      html += `</div>`;
    }
  }
  mazeGrid.innerHTML = html;
}

function startMazeTimer() {
  clearInterval(mazeTimerHandle);
  mazeSecondsLeft = mazeIntervalSeconds;
  updateMazeTimerDisplay();
  mazeTimerHandle = setInterval(() => {
    mazeSecondsLeft--;
    updateMazeTimerDisplay();
    if (mazeSecondsLeft <= 0) {
      clearInterval(mazeTimerHandle);
      openMazeQuestion();
    }
  }, 1000);
}

function updateMazeTimerDisplay() {
  mazeTimerEl.textContent = mazeSecondsLeft;
  mazeTimerEl.classList.toggle('urgent', mazeSecondsLeft <= 3);
}

function setMazeControlsEnabled(enabled) {
  [mazeUpBtn, mazeDownBtn, mazeLeftBtn, mazeRightBtn].forEach((btn) => { btn.disabled = !enabled; });
}

function handleMazeKeydown(e) {
  if (!mazeActive || mazeAwaitingAnswer) return;
  const map = { ArrowUp: 'top', ArrowDown: 'bottom', ArrowLeft: 'left', ArrowRight: 'right' };
  if (map[e.key]) {
    e.preventDefault();
    tryMazeMove(map[e.key]);
  }
}

mazeUpBtn.addEventListener('click', () => tryMazeMove('top'));
mazeDownBtn.addEventListener('click', () => tryMazeMove('bottom'));
mazeLeftBtn.addEventListener('click', () => tryMazeMove('left'));
mazeRightBtn.addEventListener('click', () => tryMazeMove('right'));

const MAZE_MOVE_DELTA = { top: { dr: -1, dc: 0 }, bottom: { dr: 1, dc: 0 }, left: { dr: 0, dc: -1 }, right: { dr: 0, dc: 1 } };

function tryMazeMove(dir) {
  if (!mazeActive || mazeAwaitingAnswer) return;
  const player = mazePath[mazePath.length - 1];
  const cell = mazeGridCells[player.r][player.c];
  if (cell.walls[dir]) return; // wall blocks this direction

  const { dr, dc } = MAZE_MOVE_DELTA[dir];
  const next = { r: player.r + dr, c: player.c + dc };
  mazePath.push(next);
  mazeXp = Math.max(15, mazeXp - 1);
  renderMazeGrid();

  if (next.r === mazeSize - 1 && next.c === mazeSize - 1) {
    finishMaze();
  }
}

// ---- Blocking question, triggered every mazeIntervalSeconds ----
function openMazeQuestion() {
  mazeAwaitingAnswer = true;
  setMazeControlsEnabled(false);

  if (mazeQuestionIndex >= mazeQuestions.length) {
    mazeQuestions = shuffleArray(mazeQuestions);
    mazeQuestionIndex = 0;
  }
  const q = mazeQuestions[mazeQuestionIndex];
  mazeQuestionIndex++;

  mazeQuestionText.textContent = q.question;
  mazeQuestionFeedback.textContent = '';
  mazeQuestionFeedback.className = 'lesson-feedback';
  mazeQuestionChoices.innerHTML = q.choices.map((choice, i) => `
    <button class="lesson-choice" data-index="${i}">${escapeHtml(choice)}</button>
  `).join('');

  mazeQuestionChoices.querySelectorAll('.lesson-choice').forEach((btn) => {
    btn.addEventListener('click', () => answerMazeQuestion(btn, q));
  });

  mazeQuestionModalOverlay.classList.add('show');
}

function answerMazeQuestion(btn, q) {
  // Lock out further taps once one choice has been picked.
  const buttons = mazeQuestionChoices.querySelectorAll('.lesson-choice');
  if (buttons[0]?.disabled) return;
  buttons.forEach((b) => { b.disabled = true; });

  const selectedIndex = Number(btn.dataset.index);
  const isCorrect = selectedIndex === q.correctIndex;
  buttons.forEach((b, i) => {
    if (i === q.correctIndex) b.classList.add('correct');
    else if (i === selectedIndex) b.classList.add('wrong');
  });

  if (isCorrect) {
    playCorrectSound();
    mazeQuestionFeedback.textContent = 'Correct! Back to the maze.';
    mazeQuestionFeedback.classList.add('correct');
    if (groupGameActive && groupGameCode && auth.currentUser) {
      reportCorrectAnswer(groupGameCode, auth.currentUser.uid).catch(() => {});
    }
  } else {
    playWrongSound();
    mazeQuestionFeedback.textContent = 'Not quite — a couple steps back.';
    mazeQuestionFeedback.classList.add('wrong');
    // Penalty: undo up to the last 2 moves (never past the start cell).
    const stepsBack = Math.min(2, mazePath.length - 1);
    mazePath.splice(mazePath.length - stepsBack, stepsBack);
    renderMazeGrid();
  }

  setTimeout(() => {
    mazeQuestionModalOverlay.classList.remove('show');
    mazeAwaitingAnswer = false;
    if (mazeActive) {
      setMazeControlsEnabled(true);
      startMazeTimer();
    }
  }, 1100);
}

async function finishMaze() {
  mazeActive = false;
  clearInterval(mazeTimerHandle);
  document.removeEventListener('keydown', handleMazeKeydown);
  mazeViewOverlay.classList.remove('show');

  if (groupGameActive) {
    showGroupGameRoundEnd(`Maze solved! Score keeps climbing — ${mazePath.length - 1} moves.`);
    return;
  }

  const moves = mazePath.length - 1;
  const elapsedSeconds = Math.max(0, Math.round((Date.now() - mazeStartTime) / 1000));
  await awardGameXp(mazeXp);

  mazeFinishedXpCount.textContent = mazeXp;
  mazeFinishedMoves.textContent = moves;
  mazeFinishedTime.textContent = formatGameTime(elapsedSeconds);
  mazeFinishedModalOverlay.classList.add('show');
  await checkGameBadge('maze');
}

mazeFinishedDoneBtn.addEventListener('click', () => {
  mazeFinishedModalOverlay.classList.remove('show');
});

mazeExitBtn.addEventListener('click', () => {
  mazeActive = false;
  mazeAwaitingAnswer = false;
  clearInterval(mazeTimerHandle);
  document.removeEventListener('keydown', handleMazeKeydown);
  mazeViewOverlay.classList.remove('show');
  mazeQuestionModalOverlay.classList.remove('show');
  if (groupGameActive) showGroupGameRoundEnd('Left the maze — the clock is still running!');
});

// ============================================================
// SEESAW — 2-player pass-and-play. The screen fills with the course color;
// it slowly drains from the top over the turn timer. Answer the question
// pinned at the bottom before it fully drains, or that player loses.
// A correct answer flips the whole screen 180° (color + text) so the
// player on the other side of the phone takes their turn right-side up.
// ============================================================

// ---- Choose modal (topic + turn timer), mirrors the trivia/maze choose flow ----
function openSeesawChooseModal() {
  seesawChooseError.textContent = '';
  seesawCustomInput.value = '';
  seesawReady = false;
  seesawQuestions = [];
  seesawGenerateBtn.disabled = false;
  seesawGenerateBtn.textContent = 'Generate';
  seesawChooseModalOverlay.querySelector('h2').textContent =
    seesawMode === 'host' ? 'Choose Your Seesaw (Hosting)' : 'Choose Your Seesaw';

  if (activeCourse) {
    seesawCourseOption.classList.remove('disabled');
    seesawCourseOptionDesc.textContent = activeCourse.description || '';
    selectSeesawTopicOption('course');
  } else {
    seesawCourseOption.classList.add('disabled');
    selectSeesawTopicOption('custom');
  }
  selectSeesawDuration(seesawChosenDuration);
  seesawChooseModalOverlay.classList.add('show');
}

seesawChooseCancelBtn.addEventListener('click', () => {
  seesawChooseModalOverlay.classList.remove('show');
});

seesawCourseOption.addEventListener('click', () => {
  if (seesawCourseOption.classList.contains('disabled')) return;
  selectSeesawTopicOption('course');
});
seesawCustomOption.addEventListener('click', () => selectSeesawTopicOption('custom'));
seesawCustomInput.addEventListener('click', (e) => e.stopPropagation());
seesawCustomInput.addEventListener('input', () => {
  selectSeesawTopicOption('custom');
  resetSeesawReadyState();
});

function selectSeesawTopicOption(which) {
  seesawCourseOption.classList.toggle('selected', which === 'course');
  seesawCustomOption.classList.toggle('selected', which === 'custom');
  if (which === 'custom') seesawCustomInput.focus();
  resetSeesawReadyState();
}

seesawDurationBtns.forEach((btn) => {
  btn.addEventListener('click', () => selectSeesawDuration(btn.dataset.duration));
});

function selectSeesawDuration(duration) {
  seesawChosenDuration = duration;
  seesawDurationBtns.forEach((btn) => btn.classList.toggle('selected', btn.dataset.duration === duration));
  resetSeesawReadyState();
}

// A changed selection after generating a set means that set no longer
// matches — fall back to needing a fresh Generate press.
function resetSeesawReadyState() {
  if (!seesawReady) return;
  seesawReady = false;
  seesawQuestions = [];
  seesawGenerateBtn.disabled = false;
  seesawGenerateBtn.textContent = 'Generate';
}

seesawGenerateBtn.addEventListener('click', async () => {
  if (seesawReady) {
    seesawChooseModalOverlay.classList.remove('show');
    if (seesawMode === 'host') {
      startSeesawHostSession();
    } else {
      gateAndStartGame(startSeesawView, 'seesaw', seesawQuestions.map((q) => q.question));
    }
    return;
  }

  const isCustom = seesawCustomOption.classList.contains('selected');
  let topic;
  if (isCustom) {
    topic = seesawCustomInput.value.trim();
    if (!topic) {
      seesawChooseError.textContent = 'Type a topic first.';
      return;
    }
  } else {
    if (!activeCourse) {
      seesawChooseError.textContent = 'Pick a course first.';
      return;
    }
    topic = `${activeCourse.title}: ${activeCourse.description}`;
  }

  seesawChooseError.textContent = '';
  seesawGenerateBtn.disabled = true;
  seesawGenerateBtn.textContent = 'Generating your questions…';

  try {
    seesawHistoryContext = await getGameHistoryContext('seesaw');
    const data = await fetchSeesawQuestions(topic, [], seesawHistoryContext);
    seesawQuestions = data.questions;
    seesawUsedQuestions = seesawQuestions.map((q) => q.question);
    seesawLastTopic = topic;
    seesawReady = true;
    seesawGenerateBtn.disabled = false;
    seesawGenerateBtn.textContent = seesawMode === 'host' ? 'Create Game' : 'Start Seesaw';
    seesawChooseModalOverlay.classList.remove('show');
    if (seesawMode === 'host') {
      startSeesawHostSession();
    } else {
      gateAndStartGame(startSeesawView, 'seesaw', seesawQuestions.map((q) => q.question));
    }
  } catch (err) {
    seesawGenerateBtn.disabled = false;
    seesawGenerateBtn.textContent = 'Generate';
    seesawChooseError.textContent = err.message || 'Could not generate questions.';
  }
});

// ---- Host flow: create an RTDB session with the just-generated question
// set, show the join code, and wait for a friend to connect. ----
async function startSeesawHostSession() {
  seesawHostWaitStatus.textContent = "They open Learning Games → Join Game and type it in.";
  seesawHostCodeDisplay.textContent = '····';
  seesawHostWaitModalOverlay.classList.add('show');

  const color = activeCourse ? activeCourse.color : '#1E6FE0';
  const baseDuration = SEESAW_DURATIONS[seesawChosenDuration];
  const hostName = (auth.currentUser?.displayName || 'Player 1').split(' ')[0];

  try {
    const code = await hostSession('seesawSessions', {
      game: 'seesaw',
      hostUid: auth.currentUser?.uid || null,
      hostName,
      color,
      baseDuration,
      questions: seesawQuestions.map((q) => ({ question: q.question, choices: q.choices, correctIndex: q.correctIndex })),
      turn: 1,
      questionIndex: -1,       // -1 = host hasn't kicked off the first question yet
      correctCount: 0,
      currentDuration: baseDuration,
      scores: { 1: 0, 2: 0 },
    });
    seesawSessionCode = code;
    seesawHostCodeDisplay.textContent = code;
    renderJoinQr(seesawHostQrCanvas, code).catch(() => {});

    seesawUnsubscribe = listenToSession('seesawSessions', code, (session) => {
      onSeesawSessionMessage(session);
      if (session && seesawMode === 'host' && !seesawActive
        && session.status === 'active' && seesawHostWaitModalOverlay.classList.contains('show')) {
        seesawHostWaitModalOverlay.classList.remove('show');
        seesawMyPlayer = 1;
        gateAndStartGame(() => startSeesawRemoteView(session), 'seesaw', seesawQuestions.map((q) => q.question));
      }
    });
  } catch (err) {
    seesawHostWaitModalOverlay.classList.remove('show');
    seesawChooseModalOverlay.classList.add('show');
    seesawChooseError.textContent = err.message || 'Could not create a game — try again.';
  }
}

seesawHostWaitCancelBtn.addEventListener('click', () => {
  seesawHostWaitModalOverlay.classList.remove('show');
  if (seesawUnsubscribe) { seesawUnsubscribe(); seesawUnsubscribe = null; }
  if (seesawSessionCode) {
    cancelSession('seesawSessions', seesawSessionCode).catch(() => {});
  }
  seesawSessionCode = null;
  seesawMode = 'local';
});

// Calls the worker for a set of multiple-choice questions on the given
// topic, passing along any already-used questions so a mid-game top-up
// doesn't repeat them, plus `history` — [{ text, daysAgo }, ...] of
// questions played in past sessions (see gameHistory.js) — so the AI prompt
// can avoid recently-played questions and only fall back to older ones if
// it runs out of fresh material.
async function fetchSeesawQuestions(topic, previousQuestions, history) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'generateSeesawQuestions', topic, previousQuestions, history }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Could not generate questions.');
  if (!Array.isArray(data.questions) || !data.questions.length) {
    throw new Error('Question generation failed — try again.');
  }
  return data;
}

function startSeesawView() {
  const color = activeCourse ? activeCourse.color : '#1E6FE0';
  seesawContainer.style.setProperty('--cc', color);
  seesawQuestions = shuffleArray(seesawQuestions);
  seesawIndex = 0;
  seesawCurrentPlayer = 1;
  seesawFetchingMore = false;
  seesawMode = 'local';
  seesawBaseDurationSeconds = SEESAW_DURATIONS[seesawChosenDuration];
  seesawDurationSeconds = seesawBaseDurationSeconds;
  seesawCorrectCount = 0;
  seesawActive = true;
  seesawGameStartTime = Date.now();
  seesawRotator.classList.remove('flipped');
  seesawViewOverlay.classList.remove('seesaw-remote');
  seesawWaitingBanner.classList.remove('show');
  seesawChoices.style.visibility = 'visible';

  seesawViewOverlay.classList.add('show');
  startSeesawTimer();
  advanceSeesawQuestion();
}

// ---- Remote (Play With a Friend) game start — driven by the RTDB session
// instead of a locally-owned question queue. Called for both host and
// guest, once the daily-limit gate passes. ----
function startSeesawRemoteView(session) {
  const color = session.color || (activeCourse ? activeCourse.color : '#1E6FE0');
  seesawContainer.style.setProperty('--cc', color);
  seesawQuestions = session.questions || seesawQuestions;
  seesawBaseDurationSeconds = session.baseDuration;
  seesawDurationSeconds = session.currentDuration ?? session.baseDuration;
  seesawCorrectCount = session.correctCount || 0;
  seesawScores = session.scores || { 1: 0, 2: 0 };
  seesawActive = true;
  seesawGameStartTime = Date.now();
  seesawLastRemoteQuestionIndex = -1;
  seesawLastRemoteTurn = null;
  seesawRotator.classList.remove('flipped');
  seesawWaitingBanner.classList.remove('show');
  seesawChoices.style.visibility = 'visible';

  seesawViewOverlay.classList.add('seesaw-remote');
  seesawViewOverlay.classList.add('show');

  if (seesawMode === 'host') {
    updateSession('seesawSessions', seesawSessionCode, {
      questionIndex: 0,
      turn: 1,
      currentDuration: seesawBaseDurationSeconds,
      turnStartedAt: Date.now(),
    }).catch(() => {});
  }

  // Render right away from whatever we already know — don't wait on the
  // RTDB round-trip for the write above (or, for the guest, on the next
  // change after their initial join snapshot) to avoid a blank screen.
  applySeesawSessionState(seesawLatestSession || session);
}

// Single entry point for every RTDB update on the active session — routes
// to a full render pass once the game view is up, and just caches the
// latest state otherwise (e.g. while still in the host waiting room).
function onSeesawSessionMessage(session) {
  seesawLatestSession = session;
  if (!session) return;
  if (session.status === 'ended') {
    if (seesawActive) finishSeesawRemote(session);
    return;
  }
  if (seesawActive) applySeesawSessionState(session);
}

// Diffs the incoming session against what's already on screen and updates
// only what changed. Unlike the pass-and-play version, remote play is two
// separate devices — there's no phone to physically flip, so both players
// always see the LIVE question (their own turn or their friend's), just
// with the choice buttons only interactive when it's actually their turn.
// A same-turn question swap (wrong answer) just refreshes the text and
// lets the already-running drain continue uninterrupted; a turn change
// restarts the timer from the shared anchor.
function applySeesawSessionState(session) {
  const questionIndex = session.questionIndex ?? -1;
  if (questionIndex < 0) return; // host hasn't kicked off the first question yet

  seesawCorrectCount = session.correctCount || 0;
  if (session.scores) seesawScores = session.scores;

  if (questionIndex === seesawLastRemoteQuestionIndex) return; // only bookkeeping changed

  const isNewTurn = session.turn !== seesawLastRemoteTurn;
  seesawLastRemoteQuestionIndex = questionIndex;
  seesawLastRemoteTurn = session.turn;
  seesawBaseDurationSeconds = session.baseDuration ?? seesawBaseDurationSeconds;
  seesawDurationSeconds = session.currentDuration ?? seesawBaseDurationSeconds;
  seesawCurrentQuestion = seesawQuestions[questionIndex];
  seesawCurrentPlayer = session.turn;

  renderSeesawQuestion();
  updateSeesawTurnUI(session.turn);

  if (isNewTurn) {
    startSeesawTimer(session.turnStartedAt || Date.now());
  }
}

// Shows/hides the "waiting for your friend" banner and choice buttons
// depending on whose turn the shared session says it is. The question text
// itself stays visible for both players either way.
function updateSeesawTurnUI(turn) {
  const myTurn = turn === seesawMyPlayer;
  seesawWaitingBanner.classList.toggle('show', !myTurn);
  seesawChoices.style.visibility = myTurn ? 'visible' : 'hidden';
  seesawPlayerTag.textContent = myTurn ? 'Your turn' : "Friend's turn";
}

// Pulls the next question, kicking off a background top-up fetch once
// we're down to the second-to-last pre-generated question so play never
// has to pause waiting on the worker. Does NOT touch the timer/fill —
// called both to start a turn and mid-turn after a wrong answer.
function advanceSeesawQuestion() {
  if (seesawIndex >= seesawQuestions.length - 2 && seesawLastTopic && !seesawFetchingMore) {
    seesawFetchingMore = true;
    fetchSeesawQuestions(seesawLastTopic, seesawUsedQuestions.slice(-40), seesawHistoryContext)
      .then((data) => {
        seesawQuestions = seesawQuestions.concat(data.questions);
        seesawUsedQuestions = seesawUsedQuestions.concat(data.questions.map((q) => q.question));
      })
      .catch(() => {}) // silent — worst case we just reshuffle what we already have, below
      .finally(() => { seesawFetchingMore = false; });
  }

  if (seesawIndex >= seesawQuestions.length) {
    // Top-up hasn't landed yet (or failed) — reshuffle the existing set
    // rather than stalling the game.
    seesawQuestions = shuffleArray(seesawQuestions);
    seesawIndex = 0;
  }

  // Avoid immediately repeating the question they just got wrong, when possible.
  if (seesawQuestions.length > 1 && seesawQuestions[seesawIndex].question === seesawCurrentQuestion?.question) {
    seesawIndex++;
    if (seesawIndex >= seesawQuestions.length) seesawIndex = 0;
  }

  seesawCurrentQuestion = seesawQuestions[seesawIndex];
  seesawIndex++;

  renderSeesawQuestion();
}

function renderSeesawQuestion() {
  const q = seesawCurrentQuestion;
  seesawPlayerTag.textContent = `Player ${seesawCurrentPlayer}`;
  seesawQuestionText.textContent = q.question;
  seesawQuestionText.style.color = '';
  seesawChoices.innerHTML = q.choices.map((choice, i) => `
    <button class="lesson-choice" data-index="${i}">${escapeHtml(choice)}</button>
  `).join('');
  seesawChoices.querySelectorAll('.lesson-choice').forEach((btn) => {
    btn.addEventListener('click', () => answerSeesawQuestion(btn));
  });
}

// The colored fill drains from 100% to 0% over the turn duration — that
// drain IS the timer. It's animated with a single CSS transition (rather
// than being stepped every tick from JS) so it drains perfectly smoothly;
// a small interval just keeps the numeric badge in the middle in sync.
// Getting a question wrong does NOT call this again — the drain keeps
// running uninterrupted underneath while a new question loads.
// `anchorTime` (optional): for remote games both devices must count down
// from the SAME instant, not from whenever their own render happened to
// run — pass the shared `turnStartedAt` from the session so a device that
// heard about the turn a beat late still lands on the same deadline.
// Defaults to now, which is exactly the old local-mode behavior.
function startSeesawTimer(anchorTime) {
  clearInterval(seesawTickHandle);
  clearTimeout(seesawLoseTimeoutHandle);

  // Snap back to full instantly, with no transition...
  seesawFill.style.transition = 'none';
  seesawFill.style.height = '100%';
  void seesawFill.offsetHeight; // force reflow so the reset above is committed before animating again

  if (seesawDurationSeconds == null) {
    seesawTimerMid.textContent = '∞';
    seesawTimerMid.classList.remove('low');
    return;
  }

  seesawStartTime = anchorTime || Date.now();
  const alreadyElapsed = Math.max(0, (Date.now() - seesawStartTime) / 1000);
  const remainingNow = Math.max(0, seesawDurationSeconds - alreadyElapsed);

  seesawTimerMid.textContent = Math.ceil(remainingNow);
  seesawTimerMid.classList.remove('low');

  // ...then let the browser smoothly animate the drain over what's left.
  seesawFill.style.transition = `height ${remainingNow}s linear`;
  seesawFill.style.height = '0%';

  seesawTickHandle = setInterval(() => {
    const elapsed = (Date.now() - seesawStartTime) / 1000;
    const remaining = Math.max(0, seesawDurationSeconds - elapsed);
    const secondsLeft = Math.ceil(remaining);
    seesawTimerMid.textContent = secondsLeft;
    seesawTimerMid.classList.toggle('low', secondsLeft <= 3);
  }, 100);

  seesawLoseTimeoutHandle = setTimeout(() => {
    clearInterval(seesawTickHandle);
    if (seesawMode === 'local') {
      loseSeesaw();
    } else {
      loseSeesawRemote();
    }
  }, Math.max(0, remainingNow * 1000));
}

function answerSeesawQuestion(btn) {
  if (!seesawActive || btn.disabled) return;
  if (seesawMode !== 'local' && seesawCurrentPlayer !== seesawMyPlayer) return; // not your turn

  const selectedIndex = Number(btn.dataset.index);
  const isCorrect = selectedIndex === seesawCurrentQuestion.correctIndex;

  if (isCorrect) {
    playCorrectSound();
    clearInterval(seesawTickHandle);
    clearTimeout(seesawLoseTimeoutHandle);
    btn.classList.add('correct');
    seesawChoices.querySelectorAll('.lesson-choice').forEach((b) => { b.disabled = true; });
    setTimeout(() => {
      if (!seesawActive) return;
      if (seesawMode === 'local') flipSeesaw(); else flipSeesawRemote();
    }, 500);
  } else {
    // Wrong — the drain keeps running uninterrupted; swap in a new question
    // right away instead of letting them keep guessing the same one.
    playWrongSound();
    btn.classList.add('wrong');
    seesawChoices.querySelectorAll('.lesson-choice').forEach((b) => { b.disabled = true; });
    setTimeout(() => {
      if (!seesawActive) return;
      if (seesawMode === 'local') advanceSeesawQuestion(); else advanceSeesawQuestionRemote();
    }, 500);
  }
}

// Every 2 total correct answers (one from each player) the turn timer
// speeds up by 1 second, down to SEESAW_MIN_DURATION. Infinite mode (no
// base duration) has nothing to speed up.
function nextSeesawDurationFor(correctCount) {
  if (seesawBaseDurationSeconds == null) return null;
  const stepsDown = Math.floor(correctCount / 2);
  return Math.max(SEESAW_MIN_DURATION, seesawBaseDurationSeconds - stepsDown);
}

// Correct answer (local pass-and-play): flip the whole screen 180° (color +
// text together) so the player on the other side of the phone is now
// right-side up, hand them a fresh full meter, and load their question.
function flipSeesaw() {
  seesawCorrectCount++;
  seesawDurationSeconds = nextSeesawDurationFor(seesawCorrectCount);
  seesawCurrentPlayer = seesawCurrentPlayer === 1 ? 2 : 1;
  seesawRotator.classList.toggle('flipped', seesawCurrentPlayer === 2);
  startSeesawTimer();
  advanceSeesawQuestion();
}

// Correct answer (remote): write the shared state forward — this device
// doesn't render the next question itself, it just becomes "not my turn"
// and waits for the RTDB echo (same path both devices use) to drive the
// render, so host and guest are always working off one source of truth.
function flipSeesawRemote() {
  const nextPlayer = seesawMyPlayer === 1 ? 2 : 1;
  const newCorrectCount = seesawCorrectCount + 1;
  const nextDuration = nextSeesawDurationFor(newCorrectCount);
  const nextIndex = pickNextSeesawIndex();
  const newScores = { 1: seesawScores[1] || 0, 2: seesawScores[2] || 0 };
  newScores[seesawMyPlayer] = (newScores[seesawMyPlayer] || 0) + 1;

  updateSeesawTurnUI(nextPlayer);
  updateSession('seesawSessions', seesawSessionCode, {
    turn: nextPlayer,
    questionIndex: nextIndex,
    currentDuration: nextDuration,
    correctCount: newCorrectCount,
    scores: newScores,
    turnStartedAt: Date.now(),
  }).catch(() => {});
}

// Wrong answer (remote): swap in a new question for the SAME player — turn,
// score, and duration are untouched, and the drain keeps running.
function advanceSeesawQuestionRemote() {
  const nextIndex = pickNextSeesawIndex();
  updateSession('seesawSessions', seesawSessionCode, { questionIndex: nextIndex }).catch(() => {});
}

function pickNextSeesawIndex() {
  let idx = seesawLastRemoteQuestionIndex + 1;
  if (idx >= seesawQuestions.length) idx = 0;
  if (seesawQuestions.length > 1 && seesawQuestions[idx]?.question === seesawCurrentQuestion?.question) {
    idx = (idx + 1) % seesawQuestions.length;
  }
  return idx;
}

async function loseSeesaw() {
  seesawActive = false;
  clearInterval(seesawTickHandle);
  clearTimeout(seesawLoseTimeoutHandle);
  seesawViewOverlay.classList.remove('show');

  const loser = `Player ${seesawCurrentPlayer}`;
  const correctAnswer = seesawCurrentQuestion?.choices?.[seesawCurrentQuestion.correctIndex] || '';
  seesawLoseIcon.textContent = 'timer_off';
  seesawLoseIcon.style.color = '#E0503A';
  seesawLoseTitle.textContent = "Time's up!";
  seesawLoseStats.textContent = correctAnswer
    ? `${loser} ran out of time. The correct answer was "${correctAnswer}".`
    : `${loser} ran out of time.`;

  const elapsedSeconds = (Date.now() - seesawGameStartTime) / 1000;
  const xp = Math.min(125, Math.floor(elapsedSeconds * 2));
  await awardGameXp(xp);
  seesawLoseXp.innerHTML = `<span class="material-symbols-outlined">bolt</span> ${xp} XP earned`;

  seesawLoseModalOverlay.classList.add('show');
}

// Timed out, remote mode: since both devices now run the drain/timer in
// sync (not just whoever's turn it is), only the device whose turn it
// actually is should report the loss — the spectator's identical timeout
// firing a beat later must be a no-op.
async function loseSeesawRemote() {
  if (!seesawActive || seesawMode === 'local') return;
  if (seesawCurrentPlayer !== seesawMyPlayer) return; // not my turn — nothing to report
  clearInterval(seesawTickHandle);
  clearTimeout(seesawLoseTimeoutHandle);
  await updateSession('seesawSessions', seesawSessionCode, {
    status: 'ended',
    endedReason: 'timeout',
    loser: seesawMyPlayer,
  }).catch(() => {});
}

// Runs on both devices once a remote session ends — shows the loser the
// familiar "time's up" screen and the winner a "you won" variant of it.
async function finishSeesawRemote(session) {
  if (!seesawActive) return;
  seesawActive = false;
  clearInterval(seesawTickHandle);
  clearTimeout(seesawLoseTimeoutHandle);
  seesawViewOverlay.classList.remove('show');
  if (seesawUnsubscribe) { seesawUnsubscribe(); seesawUnsubscribe = null; }

  const iLost = session.loser === seesawMyPlayer;
  const otherName = (seesawMyPlayer === 1 ? session.guestName : session.hostName) || 'Your friend';
  const correctAnswer = seesawCurrentQuestion?.choices?.[seesawCurrentQuestion.correctIndex] || '';

  seesawLoseIcon.textContent = iLost ? 'timer_off' : 'emoji_events';
  seesawLoseIcon.style.color = iLost ? '#E0503A' : '#2FAE66';
  seesawLoseTitle.textContent = iLost ? "Time's up!" : 'You won!';
  seesawLoseStats.textContent = iLost
    ? (correctAnswer ? `You ran out of time. The correct answer was "${correctAnswer}".` : 'You ran out of time.')
    : `${otherName} ran out of time — nice work!`;

  const elapsedSeconds = (Date.now() - seesawGameStartTime) / 1000;
  // Winner gets a flat bonus on top of the shared time-based base, so the
  // player who actually won always nets more XP than the one who lost.
  const baseXp = Math.min(125, Math.floor(elapsedSeconds * 2));
  const xp = iLost ? baseXp : baseXp + SEESAW_WINNER_XP_BONUS;
  await awardGameXp(xp);
  seesawLoseXp.innerHTML = `<span class="material-symbols-outlined">bolt</span> ${xp} XP earned`;

  seesawLoseModalOverlay.classList.add('show');
  seesawMode = 'local';
  seesawSessionCode = null;
}

seesawLoseDoneBtn.addEventListener('click', () => {
  seesawLoseModalOverlay.classList.remove('show');
});

seesawExitBtn.addEventListener('click', () => {
  seesawActive = false;
  clearInterval(seesawTickHandle);
  clearTimeout(seesawLoseTimeoutHandle);
  seesawViewOverlay.classList.remove('show');
  if (seesawMode !== 'local' && seesawSessionCode) {
    updateSession('seesawSessions', seesawSessionCode, { status: 'ended', endedReason: 'left' }).catch(() => {});
  }
  if (seesawUnsubscribe) { seesawUnsubscribe(); seesawUnsubscribe = null; }
  seesawMode = 'local';
  seesawSessionCode = null;
});

// ============================================================
// DUEL ("Who Can Answer First?") — 2 players, same device, same question,
// same time. The screen splits top/bottom (top half rotated 180° so a
// player facing the other way still reads it right-side up). Both zones
// show the identical question; whichever player taps the correct choice
// first scores the round. A wrong tap only locks out that player for the
// round — the other player can still win it. Best score after
// DUEL_TOTAL_ROUNDS rounds wins the match.
// ============================================================

gameCardDuel.addEventListener('click', () => {
  gamesPageOverlay.classList.remove('show');
  duelModeModalOverlay.classList.add('show');
});

function openDuelChooseModal() {
  duelChooseError.textContent = '';
  duelCustomInput.value = '';
  duelReady = false;
  duelQuestions = [];
  duelGenerateBtn.disabled = false;
  duelGenerateBtn.textContent = 'Generate';
  duelChooseModalOverlay.querySelector('h2').textContent =
    duelMode === 'host' ? 'Choose Your Duel (Hosting)' : 'Choose Your Duel';

  if (activeCourse) {
    duelCourseOption.classList.remove('disabled');
    duelCourseOptionDesc.textContent = activeCourse.description || '';
    selectDuelTopicOption('course');
  } else {
    duelCourseOption.classList.add('disabled');
    selectDuelTopicOption('custom');
  }
  selectDuelTimer(duelChosenTimer);
  duelChooseModalOverlay.classList.add('show');
}

duelChooseCancelBtn.addEventListener('click', () => {
  duelChooseModalOverlay.classList.remove('show');
});

duelCourseOption.addEventListener('click', () => {
  if (duelCourseOption.classList.contains('disabled')) return;
  selectDuelTopicOption('course');
});
duelCustomOption.addEventListener('click', () => selectDuelTopicOption('custom'));
duelCustomInput.addEventListener('click', (e) => e.stopPropagation());
duelCustomInput.addEventListener('input', () => {
  selectDuelTopicOption('custom');
  resetDuelReadyState();
});

function selectDuelTopicOption(which) {
  duelCourseOption.classList.toggle('selected', which === 'course');
  duelCustomOption.classList.toggle('selected', which === 'custom');
  if (which === 'custom') duelCustomInput.focus();
  resetDuelReadyState();
}

duelTimerBtns.forEach((btn) => {
  btn.addEventListener('click', () => selectDuelTimer(btn.dataset.timer));
});

function selectDuelTimer(seconds) {
  duelChosenTimer = seconds;
  duelTimerBtns.forEach((btn) => btn.classList.toggle('selected', btn.dataset.timer === seconds));
  resetDuelReadyState();
}

// Changing topic/timer after a set is already generated invalidates it —
// back Generate out to its initial state rather than starting a stale match.
function resetDuelReadyState() {
  if (!duelReady) return;
  duelReady = false;
  duelQuestions = [];
  duelGenerateBtn.disabled = false;
  duelGenerateBtn.textContent = 'Generate';
}

duelGenerateBtn.addEventListener('click', async () => {
  if (duelReady) {
    duelChooseModalOverlay.classList.remove('show');
    if (duelMode === 'host') {
      startDuelHostSession();
    } else {
      gateAndStartGame(startDuelView, 'duel', duelQuestions.map((q) => q.question));
    }
    return;
  }

  const isCustom = duelCustomOption.classList.contains('selected');
  let topic;
  if (isCustom) {
    topic = duelCustomInput.value.trim();
    if (!topic) {
      duelChooseError.textContent = 'Type a topic first.';
      return;
    }
  } else {
    if (!activeCourse) {
      duelChooseError.textContent = 'Pick a course first.';
      return;
    }
    topic = `${activeCourse.title} — ${activeCourse.description || ''}`;
  }

  duelChooseError.textContent = '';
  duelGenerateBtn.disabled = true;
  duelGenerateBtn.textContent = 'Generating your questions…';

  try {
    duelHistoryContext = await getGameHistoryContext('duel');
    const data = await fetchDuelQuestions(topic, [], duelHistoryContext);
    duelQuestions = data.questions;
    duelUsedQuestions = duelQuestions.map((q) => q.question);
    duelLastTopic = topic;
    duelReady = true;
    duelGenerateBtn.disabled = false;
    duelGenerateBtn.textContent = duelMode === 'host' ? 'Create Game' : 'Start Duel';
    duelChooseModalOverlay.classList.remove('show');
    if (duelMode === 'host') {
      startDuelHostSession();
    } else {
      gateAndStartGame(startDuelView, 'duel', duelQuestions.map((q) => q.question));
    }
  } catch (err) {
    duelGenerateBtn.disabled = false;
    duelGenerateBtn.textContent = 'Generate';
    duelChooseError.textContent = err.message || 'Could not generate questions.';
  }
});

// ---- Host flow: create an RTDB session with the just-generated question
// set, show the join code, and wait for a friend to connect. Mirrors
// startSeesawHostSession — see that function for the shared pattern. ----
async function startDuelHostSession() {
  duelHostWaitStatus.textContent = "They open Learning Games → Join Game and type it in.";
  duelHostCodeDisplay.textContent = '····';
  duelHostWaitModalOverlay.classList.add('show');

  const color = activeCourse ? activeCourse.color : '#1E6FE0';
  const durationSeconds = DUEL_TIMERS[duelChosenTimer];
  const hostName = (auth.currentUser?.displayName || 'Player 1').split(' ')[0];

  try {
    const code = await hostSession('duelSessions', {
      game: 'duel',
      hostUid: auth.currentUser?.uid || null,
      hostName,
      color,
      durationSeconds,
      totalRounds: DUEL_TOTAL_ROUNDS,
      questions: duelQuestions.map((q) => ({ question: q.question, choices: q.choices, correctIndex: q.correctIndex })),
      roundIndex: -1,      // -1 = host hasn't kicked off the first round yet
      roundNumber: 1,
      scores: { 1: 0, 2: 0 },
    });
    duelSessionCode = code;
    duelHostCodeDisplay.textContent = code;
    renderJoinQr(duelHostQrCanvas, code).catch(() => {});

    duelUnsubscribe = listenToSession('duelSessions', code, (session) => {
      onDuelSessionMessage(session);
      if (session && duelMode === 'host' && !duelActive
        && session.status === 'active' && duelHostWaitModalOverlay.classList.contains('show')) {
        duelHostWaitModalOverlay.classList.remove('show');
        duelMyPlayer = 1;
        gateAndStartGame(() => startDuelRemoteView(session), 'duel', duelQuestions.map((q) => q.question));
      }
    });
  } catch (err) {
    duelHostWaitModalOverlay.classList.remove('show');
    duelChooseModalOverlay.classList.add('show');
    duelChooseError.textContent = err.message || 'Could not create a game — try again.';
  }
}

duelHostWaitCancelBtn.addEventListener('click', () => {
  duelHostWaitModalOverlay.classList.remove('show');
  if (duelUnsubscribe) { duelUnsubscribe(); duelUnsubscribe = null; }
  if (duelSessionCode) {
    cancelSession('duelSessions', duelSessionCode).catch(() => {});
  }
  duelSessionCode = null;
  duelMode = 'local';
});

// `history` — [{ text, daysAgo }, ...] previously-played duel questions (see
// gameHistory.js) so the worker's AI prompt can avoid recent repeats.
async function fetchDuelQuestions(topic, previousQuestions, history) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'generateDuelQuestions', topic, previousQuestions, history }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Could not generate questions.');
  if (!Array.isArray(data.questions) || !data.questions.length) {
    throw new Error('Question generation failed — try again.');
  }
  return data;
}

function startDuelView() {
  const color = activeCourse ? activeCourse.color : '#1E6FE0';
  duelContainer.style.setProperty('--cc', color);
  duelQuestions = shuffleArray(duelQuestions);
  duelIndex = 0;
  duelRoundNumber = 1;
  duelScores = { 1: 0, 2: 0 };
  duelFetchingMore = false;
  duelDurationSeconds = DUEL_TIMERS[duelChosenTimer];
  duelActive = true;
  duelMode = 'local';
  duelContainer.classList.remove('duel-remote-p1', 'duel-remote-p2');
  duelScore1.textContent = '0';
  duelScore2.textContent = '0';

  duelViewOverlay.classList.add('show');
  advanceDuelRound();
}

// ---- Remote (Play With a Friend) game start — driven by the RTDB session
// instead of a locally-owned round queue. Called for both host and guest,
// once the daily-limit gate passes. Unlike Seesaw's turn-based flip, both
// players see the SAME round at the SAME time and race to answer first —
// each device only renders its own zone (see the .duel-remote-pN CSS). ----
function startDuelRemoteView(session) {
  const color = session.color || (activeCourse ? activeCourse.color : '#1E6FE0');
  duelContainer.style.setProperty('--cc', color);
  duelQuestions = session.questions || duelQuestions;
  duelDurationSeconds = session.durationSeconds || DUEL_TIMERS[duelChosenTimer];
  duelRemoteTotalRounds = session.totalRounds || DUEL_TOTAL_ROUNDS;
  duelScores = session.scores || { 1: 0, 2: 0 };
  duelRoundNumber = session.roundNumber || 1;
  duelActive = true;
  duelLastRemoteRoundIndex = -1;
  duelScore1.textContent = String(duelScores[1] || 0);
  duelScore2.textContent = String(duelScores[2] || 0);

  duelContainer.classList.remove('duel-remote-p1', 'duel-remote-p2');
  duelContainer.classList.add(duelMyPlayer === 1 ? 'duel-remote-p1' : 'duel-remote-p2');

  duelViewOverlay.classList.add('show');

  if (duelMode === 'host') {
    updateSession('duelSessions', duelSessionCode, {
      roundIndex: 0,
      roundNumber: 1,
      roundStartedAt: Date.now(),
    }).catch(() => {});
  }

  // Render right away from whatever we already know — don't wait on the
  // RTDB round-trip for the write above (or, for the guest, on the next
  // change after their initial join snapshot) to avoid a blank screen.
  applyDuelSessionState(duelLatestSession || session);
}

// Single entry point for every RTDB update on the active session — routes
// to a full render pass once the game view is up, and just caches the
// latest state otherwise (e.g. while still in the host waiting room).
function onDuelSessionMessage(session) {
  duelLatestSession = session;
  if (!session) return;
  if (session.status === 'ended') {
    if (duelActive) finishDuelRemote(session);
    return;
  }
  if (duelActive) applyDuelSessionState(session);
}

// Diffs the incoming session against what's already on screen. A new
// roundIndex means a fresh round — render the (identical, shared) question
// into this device's own zone and restart the timer from the shared
// anchor. Same round but a roundWinners entry just appeared means someone
// (maybe us) won the race for this round — resolve it locally.
function applyDuelSessionState(session) {
  const roundIndex = session.roundIndex ?? -1;
  if (roundIndex < 0) return; // host hasn't kicked off the first round yet

  if (session.scores) {
    duelScores = session.scores;
    duelScore1.textContent = String(duelScores[1] || 0);
    duelScore2.textContent = String(duelScores[2] || 0);
  }
  duelRoundNumber = session.roundNumber || duelRoundNumber;
  duelRemoteTotalRounds = session.totalRounds || duelRemoteTotalRounds;

  if (roundIndex !== duelLastRemoteRoundIndex) {
    duelLastRemoteRoundIndex = roundIndex;
    duelCurrentQuestion = duelQuestions[roundIndex];
    renderDuelRemoteRound();
    startDuelRemoteTimer(session.roundStartedAt || Date.now());
    return;
  }

  const winner = session.roundWinners ? session.roundWinners[roundIndex] : undefined;
  if (winner !== undefined && !duelRoundResolved) {
    resolveDuelRemoteRound(winner);
  }
}

// Renders the shared question into only THIS device's zone — the other
// zone stays hidden by CSS (.duel-remote-pN).
function renderDuelRemoteRound() {
  const q = duelCurrentQuestion;
  duelRoundResolved = false;
  duelRoundLabel.textContent = `Round ${duelRoundNumber} of ${duelRemoteTotalRounds}`;

  const zone = duelMyPlayer === 1 ? duelZoneTop : duelZoneBottom;
  const qText = duelMyPlayer === 1 ? duelQuestionTextTop : duelQuestionTextBottom;
  const choicesEl = duelMyPlayer === 1 ? duelChoicesTop : duelChoicesBottom;

  zone.classList.remove('locked', 'zone-won');
  qText.textContent = q.question;
  choicesEl.innerHTML = q.choices.map((choice, i) => `
    <button class="lesson-choice" data-index="${i}">${escapeHtml(choice)}</button>
  `).join('');
  choicesEl.querySelectorAll('.lesson-choice').forEach((btn) => {
    btn.addEventListener('click', () => answerDuelRemoteQuestion(btn));
  });
}

// Same drain-style countdown as local Duel, but anchored to the shared
// `turnStartedAt`-equivalent (`roundStartedAt`) so both devices count down
// from the exact same instant regardless of listener latency.
function startDuelRemoteTimer(anchorTime) {
  clearInterval(duelTickHandle);
  clearTimeout(duelRoundTimeoutHandle);

  duelStartTime = anchorTime || Date.now();
  const alreadyElapsed = Math.max(0, (Date.now() - duelStartTime) / 1000);
  const remainingNow = Math.max(0, duelDurationSeconds - alreadyElapsed);
  duelTimerMid.textContent = Math.ceil(remainingNow);
  duelTimerMid.classList.remove('low');

  duelTickHandle = setInterval(() => {
    const elapsed = (Date.now() - duelStartTime) / 1000;
    const remaining = Math.max(0, duelDurationSeconds - elapsed);
    const secondsLeft = Math.ceil(remaining);
    duelTimerMid.textContent = secondsLeft;
    duelTimerMid.classList.toggle('low', secondsLeft <= 2);
  }, 100);

  duelRoundTimeoutHandle = setTimeout(() => {
    clearInterval(duelTickHandle);
    // Nobody answered correctly in time — race to claim the timeout so
    // only one device advances the round.
    claimField('duelSessions', duelSessionCode, `roundWinners/${duelLastRemoteRoundIndex}`, 'timeout')
      .then((won) => { if (won) claimAndAdvanceDuelRound('timeout'); })
      .catch(() => {});
  }, Math.max(0, remainingNow * 1000));
}

// A correct tap tries to atomically claim this round's win — only the
// first device to run the transaction while it's still empty gets
// `committed: true` and becomes responsible for scoring + advancing.
// A wrong tap just locks this device's own zone for the round; the other
// player can still win it, same as local mode.
function answerDuelRemoteQuestion(btn) {
  if (!duelActive || duelRoundResolved || btn.disabled) return;

  const zone = duelMyPlayer === 1 ? duelZoneTop : duelZoneBottom;
  const choicesEl = duelMyPlayer === 1 ? duelChoicesTop : duelChoicesBottom;
  const selectedIndex = Number(btn.dataset.index);
  const isCorrect = selectedIndex === duelCurrentQuestion.correctIndex;

  if (isCorrect) {
    btn.classList.add('correct');
    choicesEl.querySelectorAll('.lesson-choice').forEach((b) => { b.disabled = true; });
    claimField('duelSessions', duelSessionCode, `roundWinners/${duelLastRemoteRoundIndex}`, duelMyPlayer)
      .then((won) => { if (won) claimAndAdvanceDuelRound(duelMyPlayer); })
      // If we lose the race, the winner's write drives our UI via the
      // session listener — nothing more to do here.
      .catch(() => {});
    return;
  }

  playWrongSound();
  btn.classList.add('wrong');
  choicesEl.querySelectorAll('.lesson-choice').forEach((b) => { b.disabled = true; });
  zone.classList.add('locked');
}

// Runs ONLY on the device that won the roundWinners claim (correct answer
// or timeout) — bumps the shared score immediately, then after a short
// pause (so both devices can see the round result) either ends the match
// or advances to the next round.
function claimAndAdvanceDuelRound(winner) {
  const newScores = { 1: duelScores[1] || 0, 2: duelScores[2] || 0 };
  if (winner === 1 || winner === 2) newScores[winner] = (newScores[winner] || 0) + 1;
  updateSession('duelSessions', duelSessionCode, { scores: newScores }).catch(() => {});

  setTimeout(() => {
    if (!duelActive || duelMode === 'local') return;
    if (duelRoundNumber >= duelRemoteTotalRounds) {
      updateSession('duelSessions', duelSessionCode, {
        status: 'ended', endedReason: 'complete',
      }).catch(() => {});
    } else {
      const nextIndex = pickNextDuelRemoteIndex();
      updateSession('duelSessions', duelSessionCode, {
        roundIndex: nextIndex,
        roundNumber: duelRoundNumber + 1,
        roundStartedAt: Date.now(),
      }).catch(() => {});
    }
  }, 1100);
}

function pickNextDuelRemoteIndex() {
  let idx = duelLastRemoteRoundIndex + 1;
  if (idx >= duelQuestions.length) idx = 0;
  if (duelQuestions.length > 1 && duelQuestions[idx]?.question === duelCurrentQuestion?.question) {
    idx = (idx + 1) % duelQuestions.length;
  }
  return idx;
}

// Runs on both devices as soon as a roundWinners entry appears for the
// current round — highlights the correct answer in this device's zone and,
// if this device's player won, flags the zone so the winner sees it.
function resolveDuelRemoteRound(winner) {
  duelRoundResolved = true;
  clearInterval(duelTickHandle);
  clearTimeout(duelRoundTimeoutHandle);

  const zone = duelMyPlayer === 1 ? duelZoneTop : duelZoneBottom;
  const choicesEl = duelMyPlayer === 1 ? duelChoicesTop : duelChoicesBottom;
  const correctIndex = duelCurrentQuestion.correctIndex;
  choicesEl.querySelectorAll('.lesson-choice').forEach((b, i) => {
    b.disabled = true;
    if (i === correctIndex) b.classList.add('correct');
  });

  if (winner !== 'timeout') playCorrectSound();
  if (winner === duelMyPlayer) zone.classList.add('zone-won');
}

// Runs on both devices once a remote match ends — shows each player their
// own final tally rather than the local "Player 1 / Player 2" framing.
async function finishDuelRemote(session) {
  if (!duelActive) return;
  duelActive = false;
  clearInterval(duelTickHandle);
  clearTimeout(duelRoundTimeoutHandle);
  duelViewOverlay.classList.remove('show');
  if (duelUnsubscribe) { duelUnsubscribe(); duelUnsubscribe = null; }

  const scores = session.scores || duelScores;
  const myScore = duelMyPlayer === 1 ? (scores[1] || 0) : (scores[2] || 0);
  const otherScore = duelMyPlayer === 1 ? (scores[2] || 0) : (scores[1] || 0);
  const otherName = (duelMyPlayer === 1 ? session.guestName : session.hostName) || 'Your friend';

  if (myScore === otherScore) {
    duelEndIcon.textContent = 'handshake';
    duelEndIcon.style.color = 'var(--blue-main)';
    duelEndTitle.textContent = "It's a tie!";
  } else if (myScore > otherScore) {
    duelEndIcon.textContent = 'emoji_events';
    duelEndIcon.style.color = '#FF8A2B';
    duelEndTitle.textContent = 'You win!';
  } else {
    duelEndIcon.textContent = 'emoji_events';
    duelEndIcon.style.color = '#FF8A2B';
    duelEndTitle.textContent = `${otherName} wins!`;
  }
  duelEndStats.textContent = `Final score — You: ${myScore}, ${otherName}: ${otherScore}`;

  const DUEL_COMPLETE_XP = 50;
  // Winner gets a flat bonus on top of the shared base so the win actually
  // pays off more than losing (or tying) does.
  const xpEarned = myScore > otherScore ? DUEL_COMPLETE_XP + DUEL_WINNER_XP_BONUS : DUEL_COMPLETE_XP;
  await awardGameXp(xpEarned);
  duelEndXp.innerHTML = `<span class="material-symbols-outlined">bolt</span> ${xpEarned} XP earned`;

  duelEndModalOverlay.classList.add('show');
  duelContainer.classList.remove('duel-remote-p1', 'duel-remote-p2');
  duelMode = 'local';
  duelSessionCode = null;
}

// Pulls the next question, kicking off a background top-up fetch once
// we're down to the second-to-last pre-generated question so play never
// has to pause waiting on the worker (same pattern as Seesaw).
function advanceDuelRound() {
  if (duelIndex >= duelQuestions.length - 2 && duelLastTopic && !duelFetchingMore) {
    duelFetchingMore = true;
    fetchDuelQuestions(duelLastTopic, duelUsedQuestions.slice(-40), duelHistoryContext)
      .then((data) => {
        duelQuestions = duelQuestions.concat(data.questions);
        duelUsedQuestions = duelUsedQuestions.concat(data.questions.map((q) => q.question));
      })
      .catch(() => {}) // silent — worst case we just reshuffle what we already have, below
      .finally(() => { duelFetchingMore = false; });
  }

  if (duelIndex >= duelQuestions.length) {
    duelQuestions = shuffleArray(duelQuestions);
    duelIndex = 0;
  }

  if (duelQuestions.length > 1 && duelQuestions[duelIndex].question === duelCurrentQuestion?.question) {
    duelIndex++;
    if (duelIndex >= duelQuestions.length) duelIndex = 0;
  }

  duelCurrentQuestion = duelQuestions[duelIndex];
  duelIndex++;

  renderDuelRound();
  startDuelTimer();
}

function renderDuelRound() {
  const q = duelCurrentQuestion;
  duelRoundResolved = false;
  duelRoundLabel.textContent = `Round ${duelRoundNumber} of ${DUEL_TOTAL_ROUNDS}`;

  [{ zone: duelZoneTop, qText: duelQuestionTextTop, choicesEl: duelChoicesTop, player: 1 },
   { zone: duelZoneBottom, qText: duelQuestionTextBottom, choicesEl: duelChoicesBottom, player: 2 }]
    .forEach(({ zone, qText, choicesEl, player }) => {
      zone.classList.remove('locked', 'zone-won');
      qText.textContent = q.question;
      choicesEl.innerHTML = q.choices.map((choice, i) => `
        <button class="lesson-choice" data-index="${i}">${escapeHtml(choice)}</button>
      `).join('');
      choicesEl.querySelectorAll('.lesson-choice').forEach((btn) => {
        btn.addEventListener('click', () => answerDuelQuestion(player, btn));
      });
    });
}

function startDuelTimer() {
  clearInterval(duelTickHandle);
  clearTimeout(duelRoundTimeoutHandle);

  duelStartTime = Date.now();
  duelTimerMid.textContent = duelDurationSeconds;
  duelTimerMid.classList.remove('low');

  duelTickHandle = setInterval(() => {
    const elapsed = (Date.now() - duelStartTime) / 1000;
    const remaining = Math.max(0, duelDurationSeconds - elapsed);
    const secondsLeft = Math.ceil(remaining);
    duelTimerMid.textContent = secondsLeft;
    duelTimerMid.classList.toggle('low', secondsLeft <= 2);
  }, 100);

  duelRoundTimeoutHandle = setTimeout(() => {
    clearInterval(duelTickHandle);
    resolveDuelRound(null); // nobody answered in time
  }, duelDurationSeconds * 1000);
}

function answerDuelQuestion(player, btn) {
  if (!duelActive || duelRoundResolved || btn.disabled) return;

  const zone = player === 1 ? duelZoneTop : duelZoneBottom;
  const choicesEl = player === 1 ? duelChoicesTop : duelChoicesBottom;
  const selectedIndex = Number(btn.dataset.index);
  const isCorrect = selectedIndex === duelCurrentQuestion.correctIndex;

  if (isCorrect) {
    btn.classList.add('correct');
    resolveDuelRound(player);
    return;
  }

  // Wrong tap only locks out this player for the round — the other player
  // can still race to the correct answer.
  playWrongSound();
  btn.classList.add('wrong');
  choicesEl.querySelectorAll('.lesson-choice').forEach((b) => { b.disabled = true; });
  zone.classList.add('locked');

  const otherChoicesEl = player === 1 ? duelChoicesBottom : duelChoicesTop;
  const otherLocked = [...otherChoicesEl.querySelectorAll('.lesson-choice')].every((b) => b.disabled);
  if (otherLocked) {
    // Both players are now locked out with nobody correct — no point in
    // waiting out the rest of the timer.
    clearInterval(duelTickHandle);
    clearTimeout(duelRoundTimeoutHandle);
    resolveDuelRound(null);
  }
}

// Ends the current round. `winner` is 1, 2, or null (timeout / nobody got it).
function resolveDuelRound(winner) {
  if (duelRoundResolved) return;
  duelRoundResolved = true;
  clearInterval(duelTickHandle);
  clearTimeout(duelRoundTimeoutHandle);

  const correctIndex = duelCurrentQuestion.correctIndex;
  [duelChoicesTop, duelChoicesBottom].forEach((choicesEl) => {
    choicesEl.querySelectorAll('.lesson-choice').forEach((b, i) => {
      b.disabled = true;
      if (i === correctIndex) b.classList.add('correct');
    });
  });

  if (winner) {
    playCorrectSound();
    duelScores[winner]++;
    (winner === 1 ? duelScore1 : duelScore2).textContent = duelScores[winner];
    (winner === 1 ? duelZoneTop : duelZoneBottom).classList.add('zone-won');
  }

  setTimeout(() => {
    if (!duelActive) return;
    if (duelRoundNumber >= DUEL_TOTAL_ROUNDS) {
      finishDuelMatch();
    } else {
      duelRoundNumber++;
      advanceDuelRound();
    }
  }, 1100);
}

async function finishDuelMatch() {
  duelActive = false;
  duelViewOverlay.classList.remove('show');

  const p1 = duelScores[1], p2 = duelScores[2];
  if (p1 === p2) {
    duelEndIcon.textContent = 'handshake';
    duelEndIcon.style.color = 'var(--blue-main)';
    duelEndTitle.textContent = "It's a tie!";
  } else {
    const winner = p1 > p2 ? 'Player 1' : 'Player 2';
    duelEndIcon.textContent = 'emoji_events';
    duelEndIcon.style.color = '#FF8A2B';
    duelEndTitle.textContent = `${winner} wins!`;
  }
  duelEndStats.textContent = `Final score — Player 1: ${p1}, Player 2: ${p2}`;

  const DUEL_COMPLETE_XP = 50;
  await awardGameXp(DUEL_COMPLETE_XP);
  duelEndXp.innerHTML = `<span class="material-symbols-outlined">bolt</span> ${DUEL_COMPLETE_XP} XP earned`;

  duelEndModalOverlay.classList.add('show');
}

duelEndDoneBtn.addEventListener('click', () => {
  duelEndModalOverlay.classList.remove('show');
});

duelExitBtn.addEventListener('click', () => {
  duelActive = false;
  clearInterval(duelTickHandle);
  clearTimeout(duelRoundTimeoutHandle);
  duelViewOverlay.classList.remove('show');
  if (duelMode !== 'local' && duelSessionCode) {
    updateSession('duelSessions', duelSessionCode, { status: 'ended', endedReason: 'left' }).catch(() => {});
  }
  if (duelUnsubscribe) { duelUnsubscribe(); duelUnsubscribe = null; }
  duelContainer.classList.remove('duel-remote-p1', 'duel-remote-p2');
  duelMode = 'local';
  duelSessionCode = null;
});

// ============================================================
// MELTDOWN — solo speed round. A lava thermometer sits on the right;
// correct answers cool it down a notch, wrong or timed-out answers heat it
// up, and it also creeps up on its own every second just from the clock
// running, so standing still is never safe. The per-question timer starts
// at MELTDOWN_START_SECONDS and shaves a little off every question, so the
// pace snowballs into a tense finish. No dead time — the next question
// loads right after the previous one resolves.
// ============================================================

// ---- Choose modal (topic + starting heat), mirrors the trivia/maze/seesaw choose flow ----
function openMeltdownChooseModal() {
  meltdownChooseError.textContent = '';
  meltdownCustomInput.value = '';
  meltdownReady = false;
  meltdownQuestions = [];
  meltdownGenerateBtn.disabled = false;
  meltdownGenerateBtn.textContent = 'Generate';

  if (activeCourse) {
    meltdownCourseOption.classList.remove('disabled');
    meltdownCourseOptionDesc.textContent = activeCourse.description || '';
    selectMeltdownTopicOption('course');
  } else {
    meltdownCourseOption.classList.add('disabled');
    selectMeltdownTopicOption('custom');
  }
  selectMeltdownDifficulty(meltdownChosenDifficulty);
  meltdownChooseModalOverlay.classList.add('show');
}

meltdownChooseCancelBtn.addEventListener('click', () => {
  meltdownChooseModalOverlay.classList.remove('show');
});

meltdownCourseOption.addEventListener('click', () => {
  if (meltdownCourseOption.classList.contains('disabled')) return;
  selectMeltdownTopicOption('course');
});
meltdownCustomOption.addEventListener('click', () => selectMeltdownTopicOption('custom'));
meltdownCustomInput.addEventListener('click', (e) => e.stopPropagation());
meltdownCustomInput.addEventListener('input', () => {
  selectMeltdownTopicOption('custom');
  resetMeltdownReadyState();
});

function selectMeltdownTopicOption(which) {
  meltdownCourseOption.classList.toggle('selected', which === 'course');
  meltdownCustomOption.classList.toggle('selected', which === 'custom');
  if (which === 'custom') meltdownCustomInput.focus();
  resetMeltdownReadyState();
}

meltdownDifficultyBtns.forEach((btn) => {
  btn.addEventListener('click', () => selectMeltdownDifficulty(btn.dataset.difficulty));
});

function selectMeltdownDifficulty(difficulty) {
  meltdownChosenDifficulty = difficulty;
  meltdownDifficultyBtns.forEach((btn) => btn.classList.toggle('selected', btn.dataset.difficulty === difficulty));
  resetMeltdownReadyState();
}

// A changed selection after generating a set means that set no longer
// matches — fall back to needing a fresh Generate press.
function resetMeltdownReadyState() {
  if (!meltdownReady) return;
  meltdownReady = false;
  meltdownQuestions = [];
  meltdownGenerateBtn.disabled = false;
  meltdownGenerateBtn.textContent = 'Generate';
}

meltdownGenerateBtn.addEventListener('click', async () => {
  if (meltdownReady) {
    meltdownChooseModalOverlay.classList.remove('show');
    gateAndStartGame(startMeltdownView, 'meltdown', meltdownQuestions.map((q) => q.question));
    return;
  }

  const isCustom = meltdownCustomOption.classList.contains('selected');
  let topic;
  if (isCustom) {
    topic = meltdownCustomInput.value.trim();
    if (!topic) {
      meltdownChooseError.textContent = 'Type a topic first.';
      return;
    }
  } else {
    if (!activeCourse) {
      meltdownChooseError.textContent = 'Pick a course first.';
      return;
    }
    topic = `${activeCourse.title}: ${activeCourse.description}`;
  }

  meltdownChooseError.textContent = '';
  meltdownGenerateBtn.disabled = true;
  meltdownGenerateBtn.textContent = 'Heating things up…';

  try {
    const history = await getGameHistoryContext('meltdown');
    const data = await fetchMeltdownQuestions(topic, history);
    meltdownQuestions = data.questions;
    meltdownReady = true;
    meltdownGenerateBtn.disabled = false;
    meltdownGenerateBtn.textContent = 'Start Meltdown';
    meltdownChooseModalOverlay.classList.remove('show');
    gateAndStartGame(startMeltdownView, 'meltdown', meltdownQuestions.map((q) => q.question));
  } catch (err) {
    meltdownGenerateBtn.disabled = false;
    meltdownGenerateBtn.textContent = 'Generate';
    meltdownChooseError.textContent = err.message || 'Could not generate questions.';
  }
});

// `history` — [{ text, daysAgo }, ...] previously-played meltdown questions
// (see gameHistory.js) so the worker's AI prompt can avoid recent repeats.
async function fetchMeltdownQuestions(topic, history) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'generateMeltdownQuestions', topic, history }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Could not generate questions.');
  if (!Array.isArray(data.questions) || !data.questions.length) {
    throw new Error('Question generation failed — try again.');
  }
  return data;
}

function startMeltdownView() {
  meltdownQuestions = shuffleArray(meltdownQuestions);
  meltdownIndex = 0;
  meltdownHeat = MELTDOWN_DIFFICULTY[meltdownChosenDifficulty];
  meltdownQuestionsAnswered = 0;
  meltdownStreak = 0;
  meltdownActive = true;
  meltdownGameStartTime = Date.now();

  meltdownStreakEl.textContent = '0';
  renderMeltdownThermo();
  meltdownViewOverlay.classList.add('show');
  advanceMeltdownQuestion();
}

function renderMeltdownThermo() {
  const pct = Math.max(0, Math.min(100, meltdownHeat));
  meltdownThermoFill.style.height = `${pct}%`;
  meltdownThermoFill.style.background = meltdownHeatColor(pct);
}

// Interpolates from a cool yellow to a hot red as heat climbs toward melting.
function meltdownHeatColor(pct) {
  const cool = [255, 209, 102];
  const hot = [255, 59, 48];
  const t = pct / 100;
  const r = Math.round(cool[0] + (hot[0] - cool[0]) * t);
  const g = Math.round(cool[1] + (hot[1] - cool[1]) * t);
  const b = Math.round(cool[2] + (hot[2] - cool[2]) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

// Pulls the next question and starts its timer. No blocking pause between
// questions — this is called immediately after the previous one resolves.
function advanceMeltdownQuestion() {
  if (meltdownIndex >= meltdownQuestions.length) {
    meltdownQuestions = shuffleArray(meltdownQuestions);
    meltdownIndex = 0;
  }
  // Avoid immediately repeating the question just answered, when possible.
  if (meltdownQuestions.length > 1 && meltdownQuestions[meltdownIndex].question === meltdownCurrentQuestion?.question) {
    meltdownIndex++;
    if (meltdownIndex >= meltdownQuestions.length) meltdownIndex = 0;
  }
  meltdownCurrentQuestion = meltdownQuestions[meltdownIndex];
  meltdownIndex++;

  renderMeltdownQuestion();
  startMeltdownTimer();
}

function renderMeltdownQuestion() {
  const q = meltdownCurrentQuestion;
  meltdownQuestionText.textContent = q.question;
  meltdownChoices.innerHTML = q.choices.map((choice, i) => `
    <button class="lesson-choice" data-index="${i}">${escapeHtml(choice)}</button>
  `).join('');
  meltdownChoices.querySelectorAll('.lesson-choice').forEach((btn) => {
    btn.addEventListener('click', () => answerMeltdownQuestion(btn));
  });
}

// Each question's time budget starts at MELTDOWN_START_SECONDS and shaves
// MELTDOWN_SECONDS_STEP off per question answered so far, down to a floor.
// The tick interval also does double duty: it's what drives the passive
// heat creep, and it can trigger a melt mid-question if that creep alone
// tips the bar over the top.
function startMeltdownTimer() {
  clearInterval(meltdownTickHandle);
  clearTimeout(meltdownTimeoutHandle);

  meltdownSecondsForQuestion = Math.max(
    MELTDOWN_MIN_SECONDS,
    MELTDOWN_START_SECONDS - MELTDOWN_SECONDS_STEP * meltdownQuestionsAnswered
  );
  meltdownStartTime = Date.now();
  let lastTick = meltdownStartTime;
  updateMeltdownTimerDisplay(meltdownSecondsForQuestion);

  meltdownTickHandle = setInterval(() => {
    const now = Date.now();
    const dt = (now - lastTick) / 1000;
    lastTick = now;

    const elapsed = (now - meltdownStartTime) / 1000;
    const remaining = Math.max(0, meltdownSecondsForQuestion - elapsed);
    updateMeltdownTimerDisplay(remaining);

    meltdownHeat = Math.min(MELTDOWN_HEAT_MAX, meltdownHeat + MELTDOWN_PASSIVE_HEAT_PER_SECOND * dt);
    renderMeltdownThermo();
    if (meltdownHeat >= MELTDOWN_HEAT_MAX) {
      clearInterval(meltdownTickHandle);
      clearTimeout(meltdownTimeoutHandle);
      meltdownChoices.querySelectorAll('.lesson-choice').forEach((b) => { b.disabled = true; });
      finishMeltdown();
    }
  }, 100);

  meltdownTimeoutHandle = setTimeout(() => {
    clearInterval(meltdownTickHandle);
    handleMeltdownTimeout();
  }, meltdownSecondsForQuestion * 1000);
}

function updateMeltdownTimerDisplay(seconds) {
  meltdownTimerEl.textContent = seconds.toFixed(1);
  meltdownTimerEl.classList.toggle('urgent', seconds <= 3);
}

function handleMeltdownTimeout() {
  if (!meltdownActive) return;
  playWrongSound();
  meltdownChoices.querySelectorAll('.lesson-choice').forEach((b, i) => {
    b.disabled = true;
    if (i === meltdownCurrentQuestion.correctIndex) b.classList.add('correct');
  });
  applyMeltdownWrong();
}

function answerMeltdownQuestion(btn) {
  if (!meltdownActive || btn.disabled) return;
  clearInterval(meltdownTickHandle);
  clearTimeout(meltdownTimeoutHandle);

  const buttons = meltdownChoices.querySelectorAll('.lesson-choice');
  buttons.forEach((b) => { b.disabled = true; });

  const selectedIndex = Number(btn.dataset.index);
  const isCorrect = selectedIndex === meltdownCurrentQuestion.correctIndex;
  buttons.forEach((b, i) => {
    if (i === meltdownCurrentQuestion.correctIndex) b.classList.add('correct');
    else if (i === selectedIndex) b.classList.add('wrong');
  });

  if (isCorrect) {
    playCorrectSound();
    meltdownStreak++;
    meltdownStreakEl.textContent = String(meltdownStreak);
    meltdownHeat = Math.max(0, meltdownHeat - MELTDOWN_COOL_PER_CORRECT);
    meltdownQuestionsAnswered++;
    renderMeltdownThermo();
    if (groupGameActive && groupGameCode && auth.currentUser) {
      reportCorrectAnswer(groupGameCode, auth.currentUser.uid).catch(() => {});
    }
    setTimeout(() => {
      if (meltdownActive) advanceMeltdownQuestion();
    }, 450);
  } else {
    playWrongSound();
    applyMeltdownWrong();
  }
}

// Shared by both a wrong tap and a timeout — heats the bar up, resets the
// streak, and either melts the run or moves on to the next question.
function applyMeltdownWrong() {
  meltdownStreak = 0;
  meltdownStreakEl.textContent = '0';
  meltdownHeat = Math.min(MELTDOWN_HEAT_MAX, meltdownHeat + MELTDOWN_HEAT_PER_WRONG);
  meltdownQuestionsAnswered++;
  renderMeltdownThermo();

  setTimeout(() => {
    if (!meltdownActive) return;
    if (meltdownHeat >= MELTDOWN_HEAT_MAX) {
      finishMeltdown();
    } else {
      advanceMeltdownQuestion();
    }
  }, 650);
}

async function finishMeltdown() {
  meltdownActive = false;
  clearInterval(meltdownTickHandle);
  clearTimeout(meltdownTimeoutHandle);
  meltdownViewOverlay.classList.remove('show');

  if (groupGameActive) {
    showGroupGameRoundEnd(`Melted after a streak of ${meltdownStreak}! Keep going.`);
    return;
  }

  const best = Number(localStorage.getItem(MELTDOWN_BEST_KEY) || 0);
  const isNewBest = meltdownStreak > best;
  if (isNewBest) localStorage.setItem(MELTDOWN_BEST_KEY, String(meltdownStreak));

  meltdownLoseStats.textContent = `You melted after a streak of ${meltdownStreak} correct answer${meltdownStreak === 1 ? '' : 's'}.`;
  meltdownLoseBest.textContent = isNewBest
    ? 'New best streak!'
    : `Best streak: ${Math.max(best, meltdownStreak)}.`;

  const elapsedSeconds = (Date.now() - meltdownGameStartTime) / 1000;
  const xp = Math.min(125, Math.floor(elapsedSeconds * 2));
  await awardGameXp(xp);
  meltdownLoseXp.innerHTML = `<span class="material-symbols-outlined">bolt</span> ${xp} XP earned`;

  meltdownLoseModalOverlay.classList.add('show');
}

meltdownLoseDoneBtn.addEventListener('click', () => {
  meltdownLoseModalOverlay.classList.remove('show');
});

meltdownExitBtn.addEventListener('click', () => {
  meltdownActive = false;
  clearInterval(meltdownTickHandle);
  clearTimeout(meltdownTimeoutHandle);
  meltdownViewOverlay.classList.remove('show');
  if (groupGameActive) showGroupGameRoundEnd('Left Meltdown — the clock is still running!');
});

// ============================================================
// WORD GRID — guess a topic-related 5-letter word in 6 tries, Wordle-style.
// Guess validation is deliberately lenient (any 5-letter input is accepted
// and just gets colored) — there's no dictionary check, so the only job of
// the worker call is picking a good secret word + a short, non-giveaway hint.
// ============================================================
function openWordGridChooseModal() {
  wordGridChooseError.textContent = '';
  wordGridCustomInput.value = '';
  wordGridReady = false;
  wordGridWord = '';
  wordGridGenerateBtn.disabled = false;
  wordGridGenerateBtn.textContent = 'Generate';

  if (activeCourse) {
    wordGridCourseOption.classList.remove('disabled');
    wordGridCourseOptionDesc.textContent = activeCourse.description || '';
    selectWordGridTopicOption('course');
  } else {
    wordGridCourseOption.classList.add('disabled');
    selectWordGridTopicOption('custom');
  }
  wordGridChooseModalOverlay.classList.add('show');
}

wordGridChooseCancelBtn.addEventListener('click', () => {
  wordGridChooseModalOverlay.classList.remove('show');
});

wordGridCourseOption.addEventListener('click', () => {
  if (wordGridCourseOption.classList.contains('disabled')) return;
  selectWordGridTopicOption('course');
});
wordGridCustomOption.addEventListener('click', () => selectWordGridTopicOption('custom'));
wordGridCustomInput.addEventListener('click', (e) => e.stopPropagation());
wordGridCustomInput.addEventListener('input', () => {
  selectWordGridTopicOption('custom');
  resetWordGridReadyState();
});

function selectWordGridTopicOption(which) {
  wordGridCourseOption.classList.toggle('selected', which === 'course');
  wordGridCustomOption.classList.toggle('selected', which === 'custom');
  if (which === 'custom') wordGridCustomInput.focus();
  resetWordGridReadyState();
}

// A changed selection after generating a word means that word no longer
// matches — fall back to needing a fresh Generate press.
function resetWordGridReadyState() {
  if (!wordGridReady) return;
  wordGridReady = false;
  wordGridWord = '';
  wordGridGenerateBtn.disabled = false;
  wordGridGenerateBtn.textContent = 'Generate';
}

wordGridGenerateBtn.addEventListener('click', async () => {
  if (wordGridReady) {
    wordGridChooseModalOverlay.classList.remove('show');
    gateAndStartGame(startWordGridView, 'wordGrid', [wordGridWord]);
    return;
  }

  const isCustom = wordGridCustomOption.classList.contains('selected');
  let topic;
  if (isCustom) {
    topic = wordGridCustomInput.value.trim();
    if (!topic) {
      wordGridChooseError.textContent = 'Type a topic first.';
      return;
    }
  } else {
    if (!activeCourse) {
      wordGridChooseError.textContent = 'Pick a course first.';
      return;
    }
    topic = `${activeCourse.title}: ${activeCourse.description}`;
  }

  wordGridChooseError.textContent = '';
  wordGridGenerateBtn.disabled = true;
  wordGridGenerateBtn.textContent = 'Picking a word…';

  try {
    const history = await getGameHistoryContext('wordGrid');
    const data = await fetchWordGridWord(topic, history);
    wordGridWord = data.word;
    wordGridHint = data.hint;
    wordGridTopic = topic;
    wordGridReady = true;
    wordGridGenerateBtn.disabled = false;
    wordGridGenerateBtn.textContent = 'Start';
    wordGridChooseModalOverlay.classList.remove('show');
    gateAndStartGame(startWordGridView, 'wordGrid', [wordGridWord]);
  } catch (err) {
    wordGridGenerateBtn.disabled = false;
    wordGridGenerateBtn.textContent = 'Generate';
    wordGridChooseError.textContent = err.message || 'Could not generate a word.';
  }
});

// `history` — [{ text, daysAgo }, ...] previously-played words (see
// gameHistory.js) so the worker's AI prompt can avoid recent repeats.
async function fetchWordGridWord(topic, history) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'generateWordGridWord', topic, history }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Could not generate a word.');
  if (!data.word || !/^[A-Z]{5}$/.test(data.word)) {
    throw new Error('Word generation failed — try again.');
  }
  return data;
}

function startWordGridView() {
  wordGridGuesses = [];
  wordGridCurrentGuess = '';
  wordGridKeyStates = {};
  wordGridActive = true;

  wordGridHintText.textContent = wordGridHint;
  updateWordGridAttemptsLeft();
  wordGridMessage.textContent = '';
  renderWordGridBoard();
  renderWordGridKeyboard();
  wordGridViewOverlay.classList.add('show');
}

function updateWordGridAttemptsLeft() {
  const left = WORDGRID_MAX_GUESSES - wordGridGuesses.length;
  wordGridAttemptsLeft.textContent = `${left} left`;
}

function renderWordGridBoard() {
  const rows = [];
  for (let r = 0; r < WORDGRID_MAX_GUESSES; r++) {
    const guess = wordGridGuesses[r];
    const isCurrentRow = r === wordGridGuesses.length;
    const letters = guess
      ? guess.split('')
      : isCurrentRow
        ? wordGridCurrentGuess.split('').concat(Array(WORDGRID_WORD_LENGTH - wordGridCurrentGuess.length).fill(''))
        : Array(WORDGRID_WORD_LENGTH).fill('');

    const marks = guess ? scoreWordGridGuess(guess) : null;

    const tiles = letters.map((letter, i) => {
      const cls = ['wordgrid-tile'];
      if (letter) cls.push('filled');
      if (marks) cls.push(marks[i]);
      return `<div class="${cls.join(' ')}">${escapeHtml(letter)}</div>`;
    }).join('');

    rows.push(`<div class="wordgrid-row" data-row="${r}">${tiles}</div>`);
  }
  wordGridBoard.innerHTML = rows.join('');
}

// Wordle-style two-pass scoring: greens first (locking used letters), then
// yellows for remaining letters that appear elsewhere in the secret word.
function scoreWordGridGuess(guess) {
  const secret = wordGridWord.split('');
  const result = Array(WORDGRID_WORD_LENGTH).fill('absent');
  const used = Array(WORDGRID_WORD_LENGTH).fill(false);

  for (let i = 0; i < WORDGRID_WORD_LENGTH; i++) {
    if (guess[i] === secret[i]) {
      result[i] = 'correct';
      used[i] = true;
    }
  }
  for (let i = 0; i < WORDGRID_WORD_LENGTH; i++) {
    if (result[i] === 'correct') continue;
    const idx = secret.findIndex((ch, j) => ch === guess[i] && !used[j]);
    if (idx !== -1) {
      result[i] = 'present';
      used[idx] = true;
    }
  }
  return result;
}

function renderWordGridKeyboard() {
  wordGridKeyboard.innerHTML = WORDGRID_KEY_ROWS.map((row) => `
    <div class="wordgrid-key-row">
      ${row.map((key) => {
        if (key === 'ENTER') return `<button type="button" class="wordgrid-key wide" data-key="ENTER">Enter</button>`;
        if (key === 'DEL') return `<button type="button" class="wordgrid-key wide" data-key="DEL"><span class="material-symbols-outlined">backspace</span></button>`;
        const state = wordGridKeyStates[key];
        return `<button type="button" class="wordgrid-key${state ? ' ' + state : ''}" data-key="${key}">${key}</button>`;
      }).join('')}
    </div>
  `).join('');
  wordGridKeyboard.querySelectorAll('.wordgrid-key').forEach((btn) => {
    btn.addEventListener('click', () => handleWordGridKey(btn.dataset.key));
  });
}

function handleWordGridKey(key) {
  if (!wordGridActive) return;
  if (key === 'ENTER') {
    submitWordGridGuess();
  } else if (key === 'DEL') {
    wordGridCurrentGuess = wordGridCurrentGuess.slice(0, -1);
    renderWordGridBoard();
  } else if (/^[A-Z]$/.test(key) && wordGridCurrentGuess.length < WORDGRID_WORD_LENGTH) {
    wordGridCurrentGuess += key;
    renderWordGridBoard();
  }
}

// Physical keyboard support — only live while the Word Grid view is open, so
// it never interferes with typing anywhere else in the app.
document.addEventListener('keydown', (e) => {
  if (!wordGridActive) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const key = e.key.toUpperCase();
  if (key === 'ENTER') { e.preventDefault(); handleWordGridKey('ENTER'); return; }
  if (key === 'BACKSPACE') { e.preventDefault(); handleWordGridKey('DEL'); return; }
  if (/^[A-Z]$/.test(key)) { e.preventDefault(); handleWordGridKey(key); }
});

function submitWordGridGuess() {
  if (wordGridCurrentGuess.length < WORDGRID_WORD_LENGTH) {
    wordGridMessage.textContent = 'Not enough letters.';
    const row = wordGridBoard.querySelector(`[data-row="${wordGridGuesses.length}"]`);
    if (row) {
      row.classList.remove('shake');
      void row.offsetWidth; // restart the shake animation on repeated invalid submits
      row.classList.add('shake');
    }
    return;
  }

  wordGridMessage.textContent = '';
  const guess = wordGridCurrentGuess;
  const marks = scoreWordGridGuess(guess);
  const rank = { absent: 0, present: 1, correct: 2 };
  marks.forEach((mark, i) => {
    const letter = guess[i];
    if (!wordGridKeyStates[letter] || rank[mark] > rank[wordGridKeyStates[letter]]) {
      wordGridKeyStates[letter] = mark;
    }
  });

  wordGridGuesses.push(guess);
  wordGridCurrentGuess = '';
  renderWordGridBoard();
  renderWordGridKeyboard();
  updateWordGridAttemptsLeft();

  const won = guess === wordGridWord;
  if (won) {
    playCorrectSound();
    if (groupGameActive && groupGameCode && auth.currentUser) {
      reportCorrectAnswer(groupGameCode, auth.currentUser.uid).catch(() => {});
    }
    setTimeout(() => finishWordGrid(true), 500);
  } else if (wordGridGuesses.length >= WORDGRID_MAX_GUESSES) {
    playWrongSound();
    setTimeout(() => finishWordGrid(false), 500);
  } else {
    playWrongSound();
  }
}

async function finishWordGrid(won) {
  wordGridActive = false;
  wordGridViewOverlay.classList.remove('show');

  if (groupGameActive) {
    showGroupGameRoundEnd(won ? `Got it — ${wordGridWord}! Keep going.` : `The word was ${wordGridWord}. Keep going!`);
    return;
  }

  const guessesUsed = wordGridGuesses.length;
  const xp = won ? WORDGRID_XP_BY_GUESS[guessesUsed - 1] : WORDGRID_LOSE_XP;
  await awardGameXp(xp);

  wordGridEndTitle.textContent = won ? 'You Got It! 🎉' : 'So Close!';
  wordGridEndStats.textContent = won
    ? `You guessed ${wordGridWord} in ${guessesUsed} ${guessesUsed === 1 ? 'try' : 'tries'}.`
    : `The word was ${wordGridWord}.`;
  wordGridEndXp.innerHTML = `<span class="material-symbols-outlined">bolt</span> ${xp} XP earned`;

  wordGridEndModalOverlay.classList.add('show');
}

wordGridPlayAgainBtn.addEventListener('click', async () => {
  wordGridEndModalOverlay.classList.remove('show');
  wordGridPlayAgainBtn.disabled = true;
  wordGridPlayAgainBtn.textContent = 'Picking a word…';
  try {
    const history = await getGameHistoryContext('wordGrid');
    const data = await fetchWordGridWord(wordGridTopic, history);
    wordGridWord = data.word;
    wordGridHint = data.hint;
    gateAndStartGame(startWordGridView, 'wordGrid', [wordGridWord]);
  } catch (err) {
    wordGridChooseError.textContent = err.message || 'Could not generate a word.';
    openWordGridChooseModal();
  } finally {
    wordGridPlayAgainBtn.disabled = false;
    wordGridPlayAgainBtn.textContent = 'Play Again';
  }
});

wordGridEndDoneBtn.addEventListener('click', () => {
  wordGridEndModalOverlay.classList.remove('show');
});

wordGridExitBtn.addEventListener('click', () => {
  wordGridActive = false;
  wordGridViewOverlay.classList.remove('show');
  if (groupGameActive) showGroupGameRoundEnd('Left Word Grid — the clock is still running!');
});

// ============================================================
// CONNECTORS
// ============================================================
// Combines all of a learner's courses (plus any extra topics they add) into
// a single "Connections"-style board: one group of 4 related words per
// topic, tap 4 tiles that belong together. Unlike NYT Connections' fixed
// 4x4, this scales with however many topics the player picks.

gameCardConnectors.addEventListener('click', () => {
  gamesPageOverlay.classList.remove('show');
  openConnectorsSetup();
});

connectorsSetupExitBtn.addEventListener('click', () => {
  connectorsSetupOverlay.classList.remove('show');
});

function openConnectorsSetup() {
  connectorsSetupError.textContent = '';
  connectorsCustomTopics = [];
  connectorsTopicInput.value = '';
  // Default to "combine all your courses" — the core pitch of the game —
  // while still letting the player uncheck ones they don't want in the mix.
  connectorsSelectedCourseIds = new Set(courses.map((c) => c.id));
  renderConnectorsCourseList();
  renderConnectorsTopicChips();
  selectConnectorsDifficulty(connectorsChosenDifficulty || 'medium');
  connectorsGenerateBtn.disabled = false;
  connectorsGenerateBtn.textContent = 'Generate';
  connectorsSetupOverlay.classList.add('show');
}

function renderConnectorsCourseList() {
  if (!courses.length) {
    connectorsCourseList.innerHTML = '';
    connectorsCoursesEmptyNote.style.display = 'block';
    return;
  }
  connectorsCoursesEmptyNote.style.display = 'none';
  connectorsCourseList.innerHTML = courses.map((c) => {
    const selected = connectorsSelectedCourseIds.has(c.id);
    return `
      <button type="button" class="connectors-course-item${selected ? ' selected' : ''}" data-course-id="${c.id}">
        <span class="connectors-course-color-dot" style="background:${c.color || 'var(--blue-main)'}"></span>
        <span class="connectors-course-item-title">${escapeHtml(c.title)}</span>
        <span class="material-symbols-outlined connectors-course-check">${selected ? 'check_circle' : 'radio_button_unchecked'}</span>
      </button>
    `;
  }).join('');
  connectorsCourseList.querySelectorAll('.connectors-course-item').forEach((btn) => {
    btn.addEventListener('click', () => toggleConnectorsCourse(btn.dataset.courseId));
  });
}

function toggleConnectorsCourse(courseId) {
  if (connectorsSelectedCourseIds.has(courseId)) connectorsSelectedCourseIds.delete(courseId);
  else connectorsSelectedCourseIds.add(courseId);
  connectorsSetupError.textContent = '';
  renderConnectorsCourseList();
}

function renderConnectorsTopicChips() {
  connectorsTopicChips.innerHTML = connectorsCustomTopics.map((topic, i) => `
    <span class="connectors-topic-chip">
      ${escapeHtml(topic)}
      <button type="button" class="connectors-topic-chip-remove" data-idx="${i}" aria-label="Remove topic">
        <span class="material-symbols-outlined">close</span>
      </button>
    </span>
  `).join('');
  connectorsTopicChips.querySelectorAll('.connectors-topic-chip-remove').forEach((btn) => {
    btn.addEventListener('click', () => {
      connectorsCustomTopics.splice(Number(btn.dataset.idx), 1);
      renderConnectorsTopicChips();
    });
  });
}

function addConnectorsTopic() {
  const val = connectorsTopicInput.value.trim();
  if (!val) return;
  if (connectorsCustomTopics.some((t) => t.toLowerCase() === val.toLowerCase())) {
    connectorsTopicInput.value = '';
    return;
  }
  connectorsCustomTopics.push(val);
  connectorsTopicInput.value = '';
  connectorsSetupError.textContent = '';
  renderConnectorsTopicChips();
}
connectorsAddTopicBtn.addEventListener('click', addConnectorsTopic);
connectorsTopicInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); addConnectorsTopic(); }
});

connectorsDifficultyBtns.forEach((btn) => {
  btn.addEventListener('click', () => selectConnectorsDifficulty(btn.dataset.difficulty));
});
function selectConnectorsDifficulty(difficulty) {
  connectorsChosenDifficulty = difficulty;
  connectorsDifficultyBtns.forEach((btn) => btn.classList.toggle('selected', btn.dataset.difficulty === difficulty));
  connectorsDifficultyHint.textContent = CONNECTORS_DIFFICULTY_HINTS[difficulty] || '';
}

connectorsGenerateBtn.addEventListener('click', async () => {
  const selectedCourses = courses.filter((c) => connectorsSelectedCourseIds.has(c.id));
  const topics = [
    ...selectedCourses.map((c) => ({ label: c.title, description: c.description || '', color: c.color || null })),
    ...connectorsCustomTopics.map((t) => ({ label: t, description: '', color: null })),
  ];

  if (topics.length < CONNECTORS_MIN_TOPICS) {
    connectorsSetupError.textContent = 'Pick a course or topic.';
    return;
  }
  if (topics.length > CONNECTORS_MAX_TOPICS) {
    connectorsSetupError.textContent = `Pick ${CONNECTORS_MAX_TOPICS} or fewer — 1 topic makes the sharpest board.`;
    return;
  }

  connectorsSetupError.textContent = '';
  connectorsGenerateBtn.disabled = true;
  connectorsGenerateBtn.textContent = 'Building your board…';

  try {
    const history = await getGameHistoryContext('connectors');
    const data = await fetchConnectorsPuzzle(topics, connectorsChosenDifficulty, history);
    setUpConnectorsGame(data.groups, topics);
    connectorsSetupOverlay.classList.remove('show');
    gateAndStartGame(startConnectorsView, 'connectors', connectorsGroups.flatMap((g) => g.words));
  } catch (err) {
    connectorsSetupError.textContent = err.message || 'Could not build a board. Try again.';
  } finally {
    connectorsGenerateBtn.disabled = false;
    connectorsGenerateBtn.textContent = 'Generate';
  }
});

// Calls the worker. Server contract:
//   POST { action: 'generateConnectors',
//          topics: [{ label, description }, ...],   // 1-2 topics
//          difficulty: 'easy' | 'medium' | 'hard' }
//   -> { groups: [{ label, words: [N strings] }, ...], wordsPerGroup: N }
// The board is ALWAYS exactly 4 groups (rows), matching real NYT
// Connections — that never changes. Words per group is fixed at 4 for
// 1-2 topics (worker invents extra groups to fill out to 4 when fewer
// than 4 topics are given).
// On 'hard', the worker should deliberately favor words with plausible
// overlap across groups (misdirection), matching NYT Connections' difficulty.
const CONNECTORS_GROUP_COUNT = 4;

// `history` — [{ text, daysAgo }, ...] previously-played words across past
// boards (see gameHistory.js) so the worker's AI prompt can avoid recent
// repeats when picking words for each group.
async function fetchConnectorsPuzzle(topics, difficulty, history) {
  const wordsPerGroup = Math.max(CONNECTORS_GROUP_COUNT, topics.length);
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      action: 'generateConnectors',
      topics: topics.map((t) => ({ label: t.label, description: t.description })),
      difficulty,
      history,
    }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Could not build a board.');
  if (!Array.isArray(data.groups) || data.groups.length !== CONNECTORS_GROUP_COUNT) {
    throw new Error('Board generation failed — try again.');
  }
  const allWords = [];
  for (const g of data.groups) {
    if (!Array.isArray(g.words) || g.words.length !== wordsPerGroup) {
      throw new Error('Board generation failed — try again.');
    }
    allWords.push(...g.words.map((w) => String(w).trim().toLowerCase()));
  }
  // A word reused across two groups would make the puzzle unsolvable/ambiguous.
  if (new Set(allWords).size !== allWords.length) {
    throw new Error('Board generation failed — try again.');
  }
  return data;
}

function setUpConnectorsGame(rawGroups, topics) {
  // Groups no longer map 1:1 to the selected topics (the board is always
  // exactly 4 groups regardless of topic count), so color/label always come
  // from the group itself rather than the topic at the same index.
  connectorsGroups = rawGroups.map((g, i) => ({
    id: `g${i}`,
    label: g.label || `Group ${i + 1}`,
    color: CONNECTORS_FALLBACK_COLORS[i % CONNECTORS_FALLBACK_COLORS.length],
    words: g.words,
    solved: false,
  }));

  const allTiles = connectorsGroups.flatMap((g) =>
    g.words.map((word) => ({ key: `${g.id}::${word}`, word, groupId: g.id }))
  );
  connectorsTileMap = new Map(allTiles.map((t) => [t.key, t]));
  connectorsDisplayOrder = shuffleArray(allTiles.map((t) => t.key));

  connectorsSelected = [];
  connectorsGuessedSets = [];
  connectorsMaxMistakes = CONNECTORS_MISTAKES_BY_DIFFICULTY[connectorsChosenDifficulty] || 4;
  connectorsMistakesLeft = connectorsMaxMistakes;
}

function startConnectorsView() {
  connectorsViewOverlay.classList.add('show');
  connectorsFeedback.textContent = '';
  connectorsActionsRow.style.display = '';
  connectorsContinueBtn.style.display = 'none';
  renderConnectorsSolvedBands();
  renderConnectorsGrid();
  renderConnectorsMistakeDots();
  updateConnectorsSubmitState();
}

function renderConnectorsSolvedBands() {
  const solved = connectorsGroups.filter((g) => g.solved);
  connectorsSolvedBands.innerHTML = solved.map((g) => `
    <div class="connectors-solved-band" style="background:${g.color}">
      <div class="connectors-solved-band-label">${escapeHtml(g.label)}</div>
      <div class="connectors-solved-band-words">${g.words.map(escapeHtml).join(', ')}</div>
    </div>
  `).join('');
}

function renderConnectorsGrid() {
  connectorsGrid.innerHTML = connectorsDisplayOrder.map((key) => {
    const tile = connectorsTileMap.get(key);
    const selected = connectorsSelected.includes(key);
    return `<button type="button" class="connectors-tile${selected ? ' selected' : ''}" data-key="${key}">${escapeHtml(tile.word)}</button>`;
  }).join('');
  connectorsGrid.querySelectorAll('.connectors-tile').forEach((btn) => {
    btn.addEventListener('click', () => toggleConnectorsTile(btn.dataset.key));
  });
}

function toggleConnectorsTile(key) {
  const idx = connectorsSelected.indexOf(key);
  if (idx !== -1) {
    connectorsSelected.splice(idx, 1);
  } else {
    if (connectorsSelected.length >= 4) return; // NYT Connections caps a guess at 4 tiles
    connectorsSelected.push(key);
  }
  connectorsFeedback.textContent = '';
  renderConnectorsGrid();
  updateConnectorsSubmitState();
}

function updateConnectorsSubmitState() {
  connectorsSubmitBtn.disabled = connectorsSelected.length !== 4;
  connectorsDeselectBtn.style.visibility = connectorsSelected.length ? 'visible' : 'hidden';
}

connectorsDeselectBtn.addEventListener('click', () => {
  connectorsSelected = [];
  connectorsFeedback.textContent = '';
  renderConnectorsGrid();
  updateConnectorsSubmitState();
});

connectorsShuffleBtn.addEventListener('click', () => {
  connectorsDisplayOrder = shuffleArray(connectorsDisplayOrder);
  renderConnectorsGrid();
});

connectorsSubmitBtn.addEventListener('click', () => {
  if (connectorsSelected.length !== 4) return;

  const sortedGuess = [...connectorsSelected].sort();
  const alreadyTried = connectorsGuessedSets.some((g) => g.join('|') === sortedGuess.join('|'));
  if (alreadyTried) {
    connectorsFeedback.textContent = 'Already tried that group.';
    return;
  }
  connectorsGuessedSets.push(sortedGuess);

  const groupIds = connectorsSelected.map((key) => connectorsTileMap.get(key).groupId);
  const uniqueGroupIds = new Set(groupIds);

  if (uniqueGroupIds.size === 1) {
    handleConnectorsCorrectGuess([...uniqueGroupIds][0]);
  } else {
    handleConnectorsWrongGuess(groupIds);
  }
});

function handleConnectorsCorrectGuess(groupId) {
  const group = connectorsGroups.find((g) => g.id === groupId);
  group.solved = true;
  connectorsDisplayOrder = connectorsDisplayOrder.filter((key) => connectorsTileMap.get(key).groupId !== groupId);
  connectorsSelected = [];
  connectorsFeedback.textContent = '';
  playCorrectSound();
  renderConnectorsSolvedBands();
  renderConnectorsGrid();
  updateConnectorsSubmitState();
  if (groupGameActive && groupGameCode && auth.currentUser) {
    reportCorrectAnswer(groupGameCode, auth.currentUser.uid).catch(() => {});
  }

  if (connectorsGroups.every((g) => g.solved)) {
    finishConnectors(true);
  }
}

function handleConnectorsWrongGuess(groupIds) {
  // NYT-style "one away" hint: exactly 3 of the 4 tapped words share a group.
  const counts = {};
  groupIds.forEach((id) => { counts[id] = (counts[id] || 0) + 1; });
  const oneAway = Math.max(...Object.values(counts)) === 3;

  connectorsMistakesLeft = Math.max(0, connectorsMistakesLeft - 1);
  renderConnectorsMistakeDots();
  connectorsFeedback.textContent = oneAway ? 'One away!' : 'Not quite — try again.';
  playWrongSound();

  connectorsGrid.querySelectorAll('.connectors-tile.selected').forEach((el) => {
    el.classList.add('shake');
    setTimeout(() => el.classList.remove('shake'), 400);
  });

  if (connectorsMistakesLeft <= 0) {
    setTimeout(() => finishConnectors(false), 500);
  }
}

function renderConnectorsMistakeDots() {
  connectorsMistakesDots.innerHTML = Array.from({ length: connectorsMaxMistakes }).map((_, i) => `
    <span class="connectors-mistake-dot${i >= connectorsMistakesLeft ? ' used' : ''}"></span>
  `).join('');
}

// A loss reveals the remaining (unsolved) groups right on the board so the
// learner can see what they missed, then waits for them to press Continue
// before showing the end-of-game summary modal. A win skips straight to
// the summary since there's nothing left to reveal.
async function finishConnectors(won) {
  const solvedCount = connectorsGroups.filter((g) => g.solved).length;
  const totalGroups = connectorsGroups.length;
  const mistakesMade = connectorsMaxMistakes - connectorsMistakesLeft;

  // Both a win and a loss reveal the final board (all groups solved/shown)
  // and wait for the learner to press Continue before showing the
  // end-of-game summary modal — a win is exactly when the board looks
  // best, so it shouldn't skip straight past it any more than a loss does.
  connectorsGroups.forEach((g) => { g.solved = true; });
  connectorsDisplayOrder = [];
  connectorsSelected = [];
  connectorsFeedback.textContent = '';
  renderConnectorsSolvedBands();
  renderConnectorsGrid();
  connectorsActionsRow.style.display = 'none';
  connectorsContinueBtn.style.display = 'inline-flex';
  connectorsContinueBtn.onclick = () => {
    connectorsContinueBtn.style.display = 'none';
    finalizeConnectors(won, solvedCount, totalGroups, mistakesMade);
  };
}

async function finalizeConnectors(won, solvedCount, totalGroups, mistakesMade) {
  connectorsViewOverlay.classList.remove('show');
  connectorsActionsRow.style.display = '';

  if (groupGameActive) {
    showGroupGameRoundEnd(won
      ? `Board solved! Keep going.`
      : `Found ${solvedCount} of ${totalGroups} groups. Keep going!`);
    return;
  }

  const xp = won
    ? Math.max(CONNECTORS_XP_SOLVED_MIN, CONNECTORS_XP_SOLVED_BASE - mistakesMade * CONNECTORS_XP_MISTAKE_PENALTY)
    : CONNECTORS_XP_INCOMPLETE_BASE + solvedCount * CONNECTORS_XP_PER_GROUP_SOLVED;
  await awardGameXp(xp);
  await checkGameBadge('connectors');

  connectorsEndIcon.textContent = won ? 'celebration' : 'hub';
  connectorsEndTitle.textContent = won ? 'Solved It! 🎉' : 'So Close!';
  connectorsEndStats.textContent = won
    ? `You found all ${totalGroups} groups with ${mistakesMade} mistake${mistakesMade === 1 ? '' : 's'}.`
    : `You found ${solvedCount} of ${totalGroups} groups.`;
  connectorsEndXp.innerHTML = `<span class="material-symbols-outlined">bolt</span> ${xp} XP earned`;
  connectorsEndModalOverlay.classList.add('show');
}

connectorsEndDoneBtn.addEventListener('click', () => {
  connectorsEndModalOverlay.classList.remove('show');
});

connectorsExitBtn.addEventListener('click', () => {
  connectorsViewOverlay.classList.remove('show');
  if (groupGameActive) showGroupGameRoundEnd('Left Connectors — the clock is still running!');
});