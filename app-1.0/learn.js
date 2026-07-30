// learn.js — Kids Learning Lab "Learn" page (Duolingo-style AI courses)
import { db, auth } from "./firebase.js";
import { notifyUser } from "./notifications.js";
import { checkStreakBadges, checkLessonBadges, checkGameBadge, checkCourseBadges } from "./badges.js";
import {
  doc, getDoc, setDoc, updateDoc, deleteDoc, collection, getDocs, query, orderBy, where, onSnapshot, serverTimestamp
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

const LEARN_WORKER_URL = 'https://kidslearninglabtextworker.nameless-cherry-998c.workers.dev/';
const MAX_COURSES = 5;
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
const learnCourseHome = document.getElementById('learnCourseHome');
const learnCourseTitle = document.getElementById('learnCourseTitle');
const learnCourseStatus = document.getElementById('learnCourseStatus');
const reviewWrongAnswersBtn = document.getElementById('reviewWrongAnswersBtn');
const reviewWrongAnswersLabel = document.getElementById('reviewWrongAnswersLabel');
const learnCreateFirstBtn = document.getElementById('learnCreateFirstBtn');

const learnUnitsList = document.getElementById('learnUnitsList');
const learnUnitPathView = document.getElementById('learnUnitPathView');
const unitPathBackBtn = document.getElementById('unitPathBackBtn');
const unitPathTitle = document.getElementById('unitPathTitle');
const learnLessonPath = document.getElementById('learnLessonPath');

const learnCourseReviewCard = document.getElementById('learnCourseReviewCard');
const learnCourseReviewTitle = document.getElementById('learnCourseReviewTitle');
const learnCourseReviewSub = document.getElementById('learnCourseReviewSub');
const learnCourseReviewBtn = document.getElementById('learnCourseReviewBtn');
const courseReviewErrorModalOverlay = document.getElementById('courseReviewErrorModalOverlay');
const courseReviewErrorCloseBtn = document.getElementById('courseReviewErrorCloseBtn');

const streakModalOverlay = document.getElementById('streakModalOverlay');
const streakModalCloseBtn = document.getElementById('streakModalCloseBtn');
const streakModalCount = document.getElementById('streakModalCount');
const streakModalMissed = document.getElementById('streakModalMissed');
const streakCalPrevBtn = document.getElementById('streakCalPrevBtn');
const streakCalNextBtn = document.getElementById('streakCalNextBtn');
const streakCalMonthLabel = document.getElementById('streakCalMonthLabel');
const streakCalGrid = document.getElementById('streakCalGrid');

const coursesModalOverlay = document.getElementById('coursesModalOverlay');
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

const createCourseModalOverlay = document.getElementById('createCourseModalOverlay');
const createCourseCancelBtn = document.getElementById('createCourseCancelBtn');
const createCourseInput = document.getElementById('createCourseInput');
const createCourseError = document.getElementById('createCourseError');
const createCourseSubmitBtn = document.getElementById('createCourseSubmitBtn');

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
const seesawLoseStats = document.getElementById('seesawLoseStats');
const seesawLoseXp = document.getElementById('seesawLoseXp');
const seesawLoseDoneBtn = document.getElementById('seesawLoseDoneBtn');

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

const learnNavBtn = document.querySelector('.nav-btn[data-page="learn"]');

// ---- State ----
let courses = [];              // all of the user's courses (metadata only)
let activeCourse = null;       // full course doc + id, currently shown on course home
let pendingShares = [];        // incoming course shares awaiting Accept/Decline
let unsubscribeSharedCourses = null; // live listener handle for pendingShares
let shareCourseTarget = null;  // the course currently open in the Share Course modal
let viewedUnitIndex = null;    // which unit's lesson path is currently open (null = units overview)
let learnProfile = { streak: 0, xp: 0, lastLessonDate: null, missedDaysInRow: 0, completedDates: [] };
let calendarViewDate = new Date();
let pendingLessonRef = null;   // { unitIndex, lessonIndex } chosen from path, shown in start modal
let titleFetchPromises = new Map(); // `${unitIndex}_${lessonIndex}` -> in-flight title pre-generation promise
let currentLesson = null;      // { id, ref, parts, isCourseReview, ... } currently open in lesson view
let wrongAnswers = [];         // wrong answers stored in Firestore for the active course
let wrongAnswersCourseId = null; // which course `wrongAnswers` was last loaded for
let currentPartIndex = 0;
let selectedChoiceIndex = null;
let answerLocked = false;
let lessonQuestionCount = 0;
let lessonCorrectCount = 0;
let lessonXp = 10;             // this lesson's running XP total (starts at 10, only persisted at completion)
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
let seesawFetchingMore = false;  // true while a background top-up fetch is in flight
let seesawIndex = 0;             // pointer into seesawQuestions
let seesawCurrentQuestion = null;
let seesawCurrentPlayer = 1;     // 1 or 2 — whoever is currently answering
let seesawDurationSeconds = 10;  // resolved seconds for the current game, or null if infinite
let seesawStartTime = 0;
let seesawGameStartTime = 0;     // Date.now() when the whole game began (not per-turn), for XP
let seesawTickHandle = null;
let seesawLoseTimeoutHandle = null;
let seesawActive = false;        // true once the seesaw view is open and playable

// ---- Duel ("Who Can Answer First?") state ----
const DUEL_TIMERS = { '5': 5, '8': 8, '12': 12 };
const DUEL_TOTAL_ROUNDS = 10;
let duelQuestions = [];          // [{ question, choices, correctIndex }, ...] fetched set
let duelReady = false;           // true once a set has been generated and Generate becomes Start
let duelChosenTimer = '8';
let duelDurationSeconds = 8;
let duelLastTopic = null;        // topic string reused when fetching more questions mid-match
let duelUsedQuestions = [];      // question texts already served
let duelFetchingMore = false;    // true while a background top-up fetch is in flight
let duelIndex = 0;               // pointer into duelQuestions
let duelCurrentQuestion = null;
let duelRoundNumber = 1;         // 1-based, shown as "Round X of 10"
let duelScores = { 1: 0, 2: 0 };
let duelStartTime = 0;
let duelTickHandle = null;
let duelRoundTimeoutHandle = null;
let duelActive = false;          // true once the duel view is open and playable
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

async function initLearn() {
  await loadStreak();
  await loadCourses();
  renderCourseHomeOrEmpty();
  startSharedCoursesListener();
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

export function resetLearnState() {
  hasInitialized = false;
  courses = [];
  activeCourse = null;
  learnProfile = { streak: 0, xp: 0, lastLessonDate: null, missedDaysInRow: 0, completedDates: [] };
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
  applyStreakDecay();
  learnStreakCount.textContent = learnProfile.streak;
  checkStreakBadges(learnProfile.streak);
}

// If more than 2 full days have passed since the last lesson, the streak resets to 0.
function applyStreakDecay() {
  if (!learnProfile.lastLessonDate) return;
  const last = new Date(learnProfile.lastLessonDate + 'T00:00:00');
  const today = new Date(todayStr() + 'T00:00:00');
  const daysSince = Math.round((today - last) / 86400000);
  const missed = Math.max(0, daysSince - 1);
  learnProfile.missedDaysInRow = missed;
  if (missed > 2) learnProfile.streak = 0;
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
  await setDoc(ref, learnProfile);
  learnStreakCount.textContent = learnProfile.streak;
  await checkStreakBadges(learnProfile.streak);
  return true;
}

learnStreakBtn.addEventListener('click', () => {
  streakModalCount.textContent = learnProfile.streak;
  streakModalMissed.textContent = learnProfile.missedDaysInRow > 0
    ? `${learnProfile.missedDaysInRow} day${learnProfile.missedDaysInRow > 1 ? 's' : ''} missed in a row`
    : 'No missed days';
  calendarViewDate = new Date();
  renderStreakCalendar();
  streakModalOverlay.classList.add('show');
});
streakModalCloseBtn.addEventListener('click', () => streakModalOverlay.classList.remove('show'));

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
  const today = todayStr();

  let html = '';
  for (let i = 0; i < firstWeekday; i++) {
    html += `<div class="streak-cal-day empty"></div>`;
  }
  for (let d = 1; d <= daysInMonth; d++) {
    const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
    const classes = ['streak-cal-day'];
    if (done.has(dateStr)) classes.push('done');
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
    learnUnitPathView.style.display = 'block';
    const unit = c.units[viewedUnitIndex];
    unitPathTitle.textContent = `Unit ${viewedUnitIndex + 1}: ${unit.title}`;
    renderLessonPath(viewedUnitIndex);
  } else {
    learnUnitPathView.style.display = 'none';
    learnUnitsList.style.display = 'grid';
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
  if (wrongAnswers.length > 0) {
    reviewWrongAnswersBtn.style.display = 'inline-flex';
    reviewWrongAnswersLabel.textContent = `Review Wrong Answers (${wrongAnswers.length})`;
  } else {
    reviewWrongAnswersBtn.style.display = 'none';
  }
}

// Builds a synthetic "lesson" from up to 10 randomized missed questions and
// plays it through the normal lesson view. It isn't backed by a real lesson
// doc, so finishLesson()/advanceAfterLessonCompletion() special-case it.
function startWrongAnswersReview() {
  if (!wrongAnswers.length) return;
  const picked = shuffleArray([...wrongAnswers]).slice(0, 10);
  const parts = picked.map((w) => ({
    type: 'question',
    question: w.question,
    choices: w.choices,
    correctIndex: w.correctIndex,
    _wrongAnswerDocId: w.id,
  }));
  currentLesson = {
    id: 'wrongAnswersReview',
    isWrongAnswerReview: true,
    isCourseReview: false,
    isUnitReview: false,
    lessonTitle: 'Review Wrong Answers',
    parts,
  };
  startLessonView();
}

reviewWrongAnswersBtn.addEventListener('click', startWrongAnswersReview);

// ============================================================
// CREATE COURSE
// ============================================================
learnCreateFirstBtn.addEventListener('click', openCreateCourseModal);
addCourseBtn.addEventListener('click', openCreateCourseModal);
createCourseCancelBtn.addEventListener('click', () => createCourseModalOverlay.classList.remove('show'));

function openCreateCourseModal() {
  if (courses.length >= MAX_COURSES) {
    createCourseError.textContent = `You can only have ${MAX_COURSES} courses at a time.`;
  } else {
    createCourseError.textContent = '';
    createCourseInput.value = '';
  }
  coursesModalOverlay.classList.remove('show');
  createCourseModalOverlay.classList.add('show');
}

createCourseSubmitBtn.addEventListener('click', async () => {
  const prompt = createCourseInput.value.trim();
  if (!prompt || courses.length >= MAX_COURSES) return;

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
async function generateAndSaveLesson(courseId, courseTitle, courseDescription, unit, unitIndex, lessonIndex, pregeneratedTitle) {
  const u = uid();
  const isReview = lessonIndex === LESSONS_PER_UNIT - 1;
  const previousTopics = await collectPreviousTopics(courseId, { unitIndex });

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
    }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Lesson generation failed.');

  const lessonRef = doc(db, 'users', u, 'learnCourses', courseId, 'lessons', `${unitIndex}_${lessonIndex}`);
  await setDoc(lessonRef, {
    lessonTitle: (!isReview && pregeneratedTitle) ? pregeneratedTitle : data.lessonTitle,
    parts: data.parts,
    topicsCovered: Array.isArray(data.topicsCovered) ? data.topicsCovered : [],
    isUnitReview: isReview,
    isCourseReview: false,
    status: 'available',
    createdAt: serverTimestamp(),
  });
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
  coursesList.innerHTML = courses.map((c) => `
    <div class="course-list-item${activeCourse?.id === c.id ? ' active-course' : ''}" style="--item-color:${c.color}">
      <button class="course-item-main" data-id="${c.id}">
        <div class="course-item-title">${escapeHtml(c.title)}</div>
        <div class="course-item-desc">${escapeHtml(c.description)}</div>
      </button>
      <button class="course-item-share-btn" data-id="${c.id}" aria-label="Share course">
        <span class="material-symbols-outlined">ios_share</span>
      </button>
      <button class="course-item-delete-btn" data-id="${c.id}" aria-label="Delete course">
        <span class="material-symbols-outlined">delete</span>
      </button>
    </div>
  `).join('');

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
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const c = courses.find((x) => x.id === btn.dataset.id);
      if (!c) return;

      if (btn.dataset.confirming !== 'true') {
        // First tap: reset any other button back to the trash icon, then arm this one.
        coursesList.querySelectorAll('.course-item-delete-btn.confirming').forEach((other) => {
          other.dataset.confirming = 'false';
          other.classList.remove('confirming');
          other.querySelector('.material-symbols-outlined').textContent = 'delete';
          other.setAttribute('aria-label', 'Delete course');
        });
        btn.dataset.confirming = 'true';
        btn.classList.add('confirming');
        btn.querySelector('.material-symbols-outlined').textContent = 'check_circle';
        btn.setAttribute('aria-label', `Confirm delete "${c.title}"`);
        return;
      }

      // Second tap: actually delete.
      btn.disabled = true;
      await deleteCourse(c.id);
    });
  });
}

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
  shareCourseModalSub.textContent = `Send "${course.title}" to a friend.`;
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
    row.querySelector('.share-course-send-btn').addEventListener('click', (e) => sendCourseShare(friend.uid, e.currentTarget));
    shareCourseFriendsList.appendChild(row);
  }
}

async function sendCourseShare(toUid, btn) {
  const u = uid();
  if (!u || !shareCourseTarget) return;
  btn.disabled = true;
  btn.textContent = 'Sending…';
  shareCourseError.textContent = '';
  try {
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
    btn.textContent = 'Sent';
  } catch (err) {
    console.error('Failed to share course:', err);
    btn.disabled = false;
    btn.textContent = 'Send';
    shareCourseError.textContent = err.message || 'Could not send that course.';
  }
}

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

  if (courses.length >= MAX_COURSES) {
    showSharedCourseCardError(shareId, `You can only have ${MAX_COURSES} courses at a time. Delete one to accept this.`);
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
async function openLessonStartModal(unitIndex, lessonIndex) {
  pendingLessonRef = { unitIndex, lessonIndex };
  const isReview = lessonIndex === LESSONS_PER_UNIT - 1;

  lessonStartTitle.textContent = isReview ? 'Unit Review' : `Loading Lesson Title...`;
  lessonStartSub.textContent = isReview ? '10 questions covering this whole unit.' : 'Loading…';
  lessonStartModalOverlay.classList.add('show');

  const u = uid();
  const lessonRef = doc(db, 'users', u, 'learnCourses', activeCourse.id, 'lessons', `${unitIndex}_${lessonIndex}`);
  const snap = await getDoc(lessonRef);

  // Bail out quietly if the modal was closed (or a different lesson picked)
  // while we were fetching.
  if (!pendingLessonRef || pendingLessonRef.unitIndex !== unitIndex || pendingLessonRef.lessonIndex !== lessonIndex) return;

  if (snap.exists()) {
    const data = snap.data();
    if (data.lessonTitle) lessonStartTitle.textContent = data.lessonTitle;
    if (!isReview) {
      lessonStartSub.textContent = data.status === 'completed' ? 'Completed — tap Start to do it again.' : '';
    }
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

// Generates and saves just a lesson's title, ahead of the full content —
// so tapping an unstarted lesson shows a real, specific title instead of a
// generic "Lesson N" placeholder. Deduped per lesson so repeated taps (or a
// tap right before Start) don't fire it twice. If the full lesson has
// already been generated by the time this resolves (e.g. the learner hit
// Start before this landed), it backs off rather than clobbering it.
async function fetchAndSaveLessonTitle(unitIndex, lessonIndex) {
  const key = `${unitIndex}_${lessonIndex}`;
  if (titleFetchPromises.has(key)) return titleFetchPromises.get(key);

  const promise = (async () => {
    try {
      const u = uid();
      const lessonRef = doc(db, 'users', u, 'learnCourses', activeCourse.id, 'lessons', key);
      const unit = activeCourse.units[unitIndex];
      const previousTopics = await collectPreviousTopics(activeCourse.id, { unitIndex });

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

      await setDoc(lessonRef, {
        lessonTitle: data.lessonTitle,
        status: 'title-only',
        createdAt: serverTimestamp(),
      });
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
  pendingLessonRef = null;
  lessonStartModalOverlay.classList.remove('show');
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

lessonStartBtn.addEventListener('click', async () => {
  if (!pendingLessonRef) return;
  const { unitIndex, lessonIndex } = pendingLessonRef;
  const u = uid();
  const lessonRef = doc(db, 'users', u, 'learnCourses', activeCourse.id, 'lessons', `${unitIndex}_${lessonIndex}`);

  lessonStartBtn.disabled = true;
  lessonStartCancelBtn.disabled = true;
  lessonStartBtnLabel.textContent = 'Loading lesson…';

  // If the title-only pre-generation for this lesson is still in flight,
  // let it land first so it can't race the full generation below and
  // overwrite it with just a title.
  const key = `${unitIndex}_${lessonIndex}`;
  if (titleFetchPromises.has(key)) {
    await titleFetchPromises.get(key);
  }

  let snap = await getDoc(lessonRef);
  if (!snap.exists() || !Array.isArray(snap.data().parts)) {
    lessonStartBtnLabel.textContent = 'Loading lesson...';
    startLessonProgressBar();
    try {
      const unit = activeCourse.units[unitIndex];
      const pregeneratedTitle = snap.exists() ? snap.data().lessonTitle : null;
      await generateAndSaveLesson(activeCourse.id, activeCourse.title, activeCourse.description, unit, unitIndex, lessonIndex, pregeneratedTitle);
      snap = await getDoc(lessonRef);
    } catch (err) {
      console.error('On-demand lesson generation failed:', err);
      stopLessonProgressBar();
      lessonStartBtn.disabled = false;
      lessonStartBtnLabel.textContent = 'Start';
      lessonStartCancelBtn.disabled = false;
      lessonStartSub.textContent = "Load failed, just press Start again.";
      return;
    }
    stopLessonProgressBar();
  }

  lessonStartBtn.disabled = false;
  lessonStartBtnLabel.textContent = 'Start';
  lessonStartCancelBtn.disabled = false;
  lessonStartModalOverlay.classList.remove('show');

  currentLesson = { id: snap.id, unitIndex, lessonIndex, ref: lessonRef, isCourseReview: false, ...snap.data() };
  pendingLessonRef = null;
  startLessonView();
});

// ============================================================
// LESSON VIEW (playing a regular lesson / unit review / course review)
// ============================================================
function startLessonView() {
  currentPartIndex = 0;
  lessonCorrectCount = 0;
  lessonQuestionCount = currentLesson.parts.filter((p) => p.type === 'question').length;
  lessonXp = 10;
  lessonStreakCount = 0;
  lessonAnsweredAny = false;
  lessonXpTracker.style.display = currentLesson.isWrongAnswerReview ? 'none' : 'flex';
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

  lessonFeedback.textContent = '';
  lessonFeedback.className = 'lesson-feedback';
  selectedChoiceIndex = null;
  answerLocked = false;
  lessonWhyBtn.style.display = 'none';
  missedQuestion = null;

  if (part.type === 'text') {
    lessonPartContent.innerHTML = `<div class="lesson-text-part">${escapeHtml(part.content)}</div>`;
    lessonActionBtn.textContent = 'Next';
    lessonActionBtn.disabled = false;
    lessonActionBtn.onclick = advancePart;
  } else {
    lessonPartContent.innerHTML = `
      <div class="lesson-question-text">${escapeHtml(part.question)}</div>
      ${part.choices.map((choice, i) => `<button class="lesson-choice" data-index="${i}">${escapeHtml(choice)}</button>`).join('')}
    `;
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
    if (!currentLesson.isWrongAnswerReview) {
      lessonStreakCount++;
      let gain = 20;
      if (lessonStreakCount === 3) gain = 30;
      else if (lessonStreakCount > 3) gain = 25;
      lessonXp += gain;
      lessonXpTrackerCount.textContent = lessonXp;
    }
    // Got it right this time during a review session — clear it from the bank.
    if (currentLesson.isWrongAnswerReview && part._wrongAnswerDocId) {
      removeWrongAnswer(part._wrongAnswerDocId);
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
    // Store every missed question (outside of review sessions, where it's
    // already in the bank) so it can resurface later.
    if (!currentLesson.isWrongAnswerReview) {
      saveWrongAnswer(part);
    }
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
  // Wrong-answer reviews aren't backed by a real lesson doc, and shouldn't
  // count toward the daily streak or XP — just show the score.
  if (currentLesson.isWrongAnswerReview) {
    showLessonSummary(false);
    return;
  }
  await updateDoc(currentLesson.ref, { status: 'completed', completedAt: serverTimestamp() });
  const streakExtended = await bumpStreak();
  await awardLessonXp();
  showLessonSummary(streakExtended);
  await checkLessonBadges();
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
  await setDoc(ref, learnProfile);
  await setDoc(doc(db, 'userProfiles', u), { xp: learnProfile.xp, streak: learnProfile.streak }, { merge: true }).catch(() => {});
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

function showLessonSummary(streakExtended) {
  lessonSummaryContainer.style.setProperty('--cc', activeCourse?.color || '#1E6FE0');
  lessonSummaryScore.textContent = `${lessonCorrectCount}/${lessonQuestionCount}`;
  if (currentLesson.isWrongAnswerReview) {
    lessonSummaryXp.style.display = 'none';
  } else {
    lessonSummaryXpCount.textContent = lessonXp;
    lessonSummaryXp.style.display = 'flex';
  }
  if (streakExtended) {
    lessonSummaryStreakCount.textContent = learnProfile.streak;
    lessonSummaryStreak.style.display = '';
  } else {
    lessonSummaryStreak.style.display = 'none';
  }
  lessonViewOverlay.classList.remove('show');
  lessonSummaryModalOverlay.classList.add('show');
}

lessonSummaryCompleteBtn.addEventListener('click', () => {
  lessonSummaryModalOverlay.classList.remove('show');
  advanceAfterLessonCompletion();
});

async function advanceAfterLessonCompletion() {
  const u = uid();

  // ---- Wrong-answer review finished: nothing to advance, just return home ----
  if (currentLesson.isWrongAnswerReview) {
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
  openSeesawChooseModal();
});
gameCardMeltdown.addEventListener('click', () => {
  gamesPageOverlay.classList.remove('show');
  openMeltdownChooseModal();
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
    if (triviaVoiceMode) startVoiceTriviaView(); else startTriviaView();
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
    const data = await fetchTrivia(topic);
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
async function fetchTrivia(topic) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'generateTrivia', topic }),
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
    const data = await fetchTrivia(triviaLastTopic);
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
      const data = await fetchTrivia(triviaLastTopic);
      if (mySession !== vtSessionId) return;
      triviaQuestions = data.questions;
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
    startMazeView();
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
    const data = await fetchMazeQuestions(topic);
    mazeQuestions = data.questions;
    mazeReady = true;
    mazeGenerateBtn.disabled = false;
    mazeGenerateBtn.textContent = 'Start Maze';
    mazeChooseModalOverlay.classList.remove('show');
    startMazeView();
  } catch (err) {
    mazeGenerateBtn.disabled = false;
    mazeGenerateBtn.textContent = 'Generate';
    mazeChooseError.textContent = err.message || 'Could not generate maze questions.';
  }
});

async function fetchMazeQuestions(topic) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'generateMazeQuestions', topic }),
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
    startSeesawView();
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
    const data = await fetchSeesawQuestions(topic, []);
    seesawQuestions = data.questions;
    seesawUsedQuestions = seesawQuestions.map((q) => q.question);
    seesawLastTopic = topic;
    seesawReady = true;
    seesawGenerateBtn.disabled = false;
    seesawGenerateBtn.textContent = 'Start Seesaw';
    seesawChooseModalOverlay.classList.remove('show');
    startSeesawView();
  } catch (err) {
    seesawGenerateBtn.disabled = false;
    seesawGenerateBtn.textContent = 'Generate';
    seesawChooseError.textContent = err.message || 'Could not generate questions.';
  }
});

// Calls the worker for a set of multiple-choice questions on the given
// topic, passing along any already-used questions so a mid-game top-up
// doesn't repeat them.
async function fetchSeesawQuestions(topic, previousQuestions) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'generateSeesawQuestions', topic, previousQuestions }),
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
  seesawDurationSeconds = SEESAW_DURATIONS[seesawChosenDuration];
  seesawActive = true;
  seesawGameStartTime = Date.now();
  seesawRotator.classList.remove('flipped');

  seesawViewOverlay.classList.add('show');
  startSeesawTimer();
  advanceSeesawQuestion();
}

// Pulls the next question, kicking off a background top-up fetch once
// we're down to the second-to-last pre-generated question so play never
// has to pause waiting on the worker. Does NOT touch the timer/fill —
// called both to start a turn and mid-turn after a wrong answer.
function advanceSeesawQuestion() {
  if (seesawIndex >= seesawQuestions.length - 2 && seesawLastTopic && !seesawFetchingMore) {
    seesawFetchingMore = true;
    fetchSeesawQuestions(seesawLastTopic, seesawUsedQuestions.slice(-40))
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
function startSeesawTimer() {
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

  seesawStartTime = Date.now();
  seesawTimerMid.textContent = seesawDurationSeconds;
  seesawTimerMid.classList.remove('low');

  // ...then let the browser smoothly animate the drain over the full duration.
  seesawFill.style.transition = `height ${seesawDurationSeconds}s linear`;
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
    loseSeesaw();
  }, seesawDurationSeconds * 1000);
}

function answerSeesawQuestion(btn) {
  if (!seesawActive || btn.disabled) return;

  const selectedIndex = Number(btn.dataset.index);
  const isCorrect = selectedIndex === seesawCurrentQuestion.correctIndex;

  if (isCorrect) {
    playCorrectSound();
    clearInterval(seesawTickHandle);
    clearTimeout(seesawLoseTimeoutHandle);
    btn.classList.add('correct');
    seesawChoices.querySelectorAll('.lesson-choice').forEach((b) => { b.disabled = true; });
    setTimeout(() => {
      if (seesawActive) flipSeesaw();
    }, 500);
  } else {
    // Wrong — the drain keeps running uninterrupted; swap in a new question
    // right away instead of letting them keep guessing the same one.
    playWrongSound();
    btn.classList.add('wrong');
    seesawChoices.querySelectorAll('.lesson-choice').forEach((b) => { b.disabled = true; });
    setTimeout(() => {
      if (seesawActive) advanceSeesawQuestion();
    }, 500);
  }
}

// Correct answer: flip the whole screen 180° (color + text together) so
// the player on the other side of the phone is now right-side up, hand
// them a fresh full meter, and load their question.
function flipSeesaw() {
  seesawCurrentPlayer = seesawCurrentPlayer === 1 ? 2 : 1;
  seesawRotator.classList.toggle('flipped', seesawCurrentPlayer === 2);
  startSeesawTimer();
  advanceSeesawQuestion();
}

async function loseSeesaw() {
  seesawActive = false;
  clearInterval(seesawTickHandle);
  clearTimeout(seesawLoseTimeoutHandle);
  seesawViewOverlay.classList.remove('show');

  const loser = `Player ${seesawCurrentPlayer}`;
  const correctAnswer = seesawCurrentQuestion?.choices?.[seesawCurrentQuestion.correctIndex] || '';
  seesawLoseStats.textContent = correctAnswer
    ? `${loser} ran out of time. The correct answer was "${correctAnswer}".`
    : `${loser} ran out of time.`;

  const elapsedSeconds = (Date.now() - seesawGameStartTime) / 1000;
  const xp = Math.min(125, Math.floor(elapsedSeconds * 2));
  await awardGameXp(xp);
  seesawLoseXp.innerHTML = `<span class="material-symbols-outlined">bolt</span> ${xp} XP earned`;

  seesawLoseModalOverlay.classList.add('show');
}

seesawLoseDoneBtn.addEventListener('click', () => {
  seesawLoseModalOverlay.classList.remove('show');
});

seesawExitBtn.addEventListener('click', () => {
  seesawActive = false;
  clearInterval(seesawTickHandle);
  clearTimeout(seesawLoseTimeoutHandle);
  seesawViewOverlay.classList.remove('show');
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
  openDuelChooseModal();
});

function openDuelChooseModal() {
  duelChooseError.textContent = '';
  duelCustomInput.value = '';
  duelReady = false;
  duelQuestions = [];
  duelGenerateBtn.disabled = false;
  duelGenerateBtn.textContent = 'Generate';

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
    startDuelView();
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
    const data = await fetchDuelQuestions(topic, []);
    duelQuestions = data.questions;
    duelUsedQuestions = duelQuestions.map((q) => q.question);
    duelLastTopic = topic;
    duelReady = true;
    duelGenerateBtn.disabled = false;
    duelGenerateBtn.textContent = 'Start Duel';
    duelChooseModalOverlay.classList.remove('show');
    startDuelView();
  } catch (err) {
    duelGenerateBtn.disabled = false;
    duelGenerateBtn.textContent = 'Generate';
    duelChooseError.textContent = err.message || 'Could not generate questions.';
  }
});

async function fetchDuelQuestions(topic, previousQuestions) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'generateDuelQuestions', topic, previousQuestions }),
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
  duelScore1.textContent = '0';
  duelScore2.textContent = '0';

  duelViewOverlay.classList.add('show');
  advanceDuelRound();
}

// Pulls the next question, kicking off a background top-up fetch once
// we're down to the second-to-last pre-generated question so play never
// has to pause waiting on the worker (same pattern as Seesaw).
function advanceDuelRound() {
  if (duelIndex >= duelQuestions.length - 2 && duelLastTopic && !duelFetchingMore) {
    duelFetchingMore = true;
    fetchDuelQuestions(duelLastTopic, duelUsedQuestions.slice(-40))
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
    startMeltdownView();
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
    const data = await fetchMeltdownQuestions(topic);
    meltdownQuestions = data.questions;
    meltdownReady = true;
    meltdownGenerateBtn.disabled = false;
    meltdownGenerateBtn.textContent = 'Start Meltdown';
    meltdownChooseModalOverlay.classList.remove('show');
    startMeltdownView();
  } catch (err) {
    meltdownGenerateBtn.disabled = false;
    meltdownGenerateBtn.textContent = 'Generate';
    meltdownChooseError.textContent = err.message || 'Could not generate questions.';
  }
});

async function fetchMeltdownQuestions(topic) {
  const res = await fetch(LEARN_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'generateMeltdownQuestions', topic }),
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
});