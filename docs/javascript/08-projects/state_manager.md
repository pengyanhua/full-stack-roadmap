# state manager.js

::: info 文件信息
- 📄 原文件：`04_state_manager.js`
- 🔤 语言：javascript
:::

============================================================
               简单状态管理器
============================================================
一个类似 Redux 的轻量级状态管理实现。
功能：
- 单一状态树
- 不可变状态更新
- Action/Reducer 模式
- 中间件支持
- 订阅机制
- 时间旅行调试
============================================================

## 完整代码

```javascript
/**
 * ============================================================
 *                简单状态管理器
 * ============================================================
 * 一个类似 Redux 的轻量级状态管理实现。
 *
 * 功能：
 * - 单一状态树
 * - 不可变状态更新
 * - Action/Reducer 模式
 * - 中间件支持
 * - 订阅机制
 * - 时间旅行调试
 * ============================================================
 */

// ============================================================
//                    核心 Store 实现
// ============================================================

/**
 * 创建 Store
 * @param {Function} reducer - 根 reducer
 * @param {any} initialState - 初始状态
 * @param {Function[]} middlewares - 中间件数组
 * @returns {Object} store
 */
function createStore(reducer, initialState, middlewares = []) {
    let state = initialState;
    let listeners = [];
    let isDispatching = false;

    // 历史记录（用于时间旅行）
    const history = [];
    let historyIndex = -1;

    /**
     * 获取当前状态
     */
    function getState() {
        if (isDispatching) {
            throw new Error("Cannot get state while dispatching");
        }
        return state;
    }

    /**
     * 订阅状态变化
     */
    function subscribe(listener) {
        if (typeof listener !== "function") {
            throw new Error("Listener must be a function");
        }

        let isSubscribed = true;
        listeners.push(listener);

        return function unsubscribe() {
            if (!isSubscribed) return;
            isSubscribed = false;
            const index = listeners.indexOf(listener);
            listeners.splice(index, 1);
        };
    }

    /**
     * 分发 action
     */
    function dispatch(action) {
        if (typeof action.type === "undefined") {
            throw new Error("Action must have a type");
        }

        if (isDispatching) {
            throw new Error("Reducers cannot dispatch actions");
        }

        try {
            isDispatching = true;
            const previousState = state;
            state = reducer(state, action);

            // 记录历史
            if (historyIndex < history.length - 1) {
                history.splice(historyIndex + 1);
            }
            history.push({ state: previousState, action });
            historyIndex = history.length - 1;

        } finally {
            isDispatching = false;
        }

        // 通知订阅者
        for (const listener of listeners) {
            listener();
        }

        return action;
    }

    /**
     * 替换 reducer
     */
    function replaceReducer(nextReducer) {
        reducer = nextReducer;
        dispatch({ type: "@@REPLACE" });
    }

    /**
     * 时间旅行 - 回到过去
     */
    function undo() {
        if (historyIndex > 0) {
            historyIndex--;
            state = history[historyIndex].state;
            for (const listener of listeners) {
                listener();
            }
        }
    }

    /**
     * 时间旅行 - 重做
     */
    function redo() {
        if (historyIndex < history.length - 1) {
            historyIndex++;
            const nextEntry = history[historyIndex + 1];
            if (nextEntry) {
                state = reducer(state, nextEntry.action);
            }
            for (const listener of listeners) {
                listener();
            }
        }
    }

    // 应用中间件
    let enhancedDispatch = dispatch;
    if (middlewares.length > 0) {
        const middlewareAPI = {
            getState,
            dispatch: (action) => enhancedDispatch(action)
        };
        const chain = middlewares.map(middleware => middleware(middlewareAPI));
        enhancedDispatch = compose(...chain)(dispatch);
    }

    // 初始化状态
    enhancedDispatch({ type: "@@INIT" });

    return {
        getState,
        subscribe,
        dispatch: enhancedDispatch,
        replaceReducer,
        undo,
        redo,
        getHistory: () => history.slice(0, historyIndex + 1)
    };
}

// ============================================================
//                    工具函数
// ============================================================

/**
 * 组合多个函数
 */
function compose(...funcs) {
    if (funcs.length === 0) {
        return (arg) => arg;
    }
    if (funcs.length === 1) {
        return funcs[0];
    }
    return funcs.reduce((a, b) => (...args) => a(b(...args)));
}

/**
 * 合并多个 reducer
 * @param {Object} reducers - reducer 对象
 * @returns {Function} 合并后的 reducer
 */
function combineReducers(reducers) {
    const reducerKeys = Object.keys(reducers);

    return function combination(state = {}, action) {
        let hasChanged = false;
        const nextState = {};

        for (const key of reducerKeys) {
            const reducer = reducers[key];
            const previousStateForKey = state[key];
            const nextStateForKey = reducer(previousStateForKey, action);

            nextState[key] = nextStateForKey;
            hasChanged = hasChanged || nextStateForKey !== previousStateForKey;
        }

        return hasChanged ? nextState : state;
    };
}

/**
 * 创建 action creator
 */
function createAction(type, payloadCreator = (x) => x) {
    const actionCreator = (...args) => ({
        type,
        payload: payloadCreator(...args)
    });
    actionCreator.type = type;
    actionCreator.toString = () => type;
    return actionCreator;
}

/**
 * 创建 reducer
 */
function createReducer(initialState, handlers) {
    return function reducer(state = initialState, action) {
        if (handlers.hasOwnProperty(action.type)) {
            return handlers[action.type](state, action);
        }
        return state;
    };
}

// ============================================================
//                    中间件
// ============================================================

/**
 * 日志中间件
 */
const loggerMiddleware = (store) => (next) => (action) => {
    console.log("dispatching:", action);
    const result = next(action);
    console.log("next state:", store.getState());
    return result;
};

/**
 * Thunk 中间件（支持异步 action）
 */
const thunkMiddleware = (store) => (next) => (action) => {
    if (typeof action === "function") {
        return action(store.dispatch, store.getState);
    }
    return next(action);
};

/**
 * 错误处理中间件
 */
const crashReporter = (store) => (next) => (action) => {
    try {
        return next(action);
    } catch (err) {
        console.error("Caught an exception!", err);
        throw err;
    }
};

// ============================================================
//                    选择器
// ============================================================

/**
 * 创建带缓存的选择器
 */
function createSelector(...args) {
    const resultFunc = args.pop();
    const selectors = args;

    let lastArgs = null;
    let lastResult = null;

    return (state) => {
        const currentArgs = selectors.map(selector => selector(state));

        // 检查参数是否变化
        const argsChanged = lastArgs === null ||
            currentArgs.some((arg, i) => arg !== lastArgs[i]);

        if (argsChanged) {
            lastResult = resultFunc(...currentArgs);
            lastArgs = currentArgs;
        }

        return lastResult;
    };
}

// ============================================================
//                    示例应用
// ============================================================

console.log("=".repeat(60));
console.log("状态管理器示例 - Todo 应用");
console.log("=".repeat(60));

// --- Action Types ---
const ActionTypes = {
    ADD_TODO: "ADD_TODO",
    TOGGLE_TODO: "TOGGLE_TODO",
    REMOVE_TODO: "REMOVE_TODO",
    SET_FILTER: "SET_FILTER"
};

// --- Action Creators ---
const addTodo = createAction(ActionTypes.ADD_TODO, (text) => ({
    id: Date.now(),
    text,
    completed: false
}));

const toggleTodo = createAction(ActionTypes.TOGGLE_TODO, (id) => ({ id }));
const removeTodo = createAction(ActionTypes.REMOVE_TODO, (id) => ({ id }));
const setFilter = createAction(ActionTypes.SET_FILTER, (filter) => ({ filter }));

// --- Reducers ---
const todosReducer = createReducer([], {
    [ActionTypes.ADD_TODO]: (state, action) => [
        ...state,
        action.payload
    ],
    [ActionTypes.TOGGLE_TODO]: (state, action) =>
        state.map(todo =>
            todo.id === action.payload.id
                ? { ...todo, completed: !todo.completed }
                : todo
        ),
    [ActionTypes.REMOVE_TODO]: (state, action) =>
        state.filter(todo => todo.id !== action.payload.id)
});

const filterReducer = createReducer("ALL", {
    [ActionTypes.SET_FILTER]: (state, action) => action.payload.filter
});

const rootReducer = combineReducers({
    todos: todosReducer,
    filter: filterReducer
});

// --- Selectors ---
const selectTodos = (state) => state.todos;
const selectFilter = (state) => state.filter;

const selectFilteredTodos = createSelector(
    selectTodos,
    selectFilter,
    (todos, filter) => {
        switch (filter) {
            case "COMPLETED":
                return todos.filter(t => t.completed);
            case "ACTIVE":
                return todos.filter(t => !t.completed);
            default:
                return todos;
        }
    }
);

// --- 创建 Store ---
const store = createStore(
    rootReducer,
    { todos: [], filter: "ALL" },
    [thunkMiddleware]  // 可添加 loggerMiddleware 查看日志
);

// --- 订阅变化 ---
const unsubscribe = store.subscribe(() => {
    const state = store.getState();
    const filtered = selectFilteredTodos(state);
    console.log("\n当前状态:");
    console.log(`  筛选: ${state.filter}`);
    console.log(`  待办: ${filtered.length} 项`);
    filtered.forEach(todo => {
        const status = todo.completed ? "✓" : "○";
        console.log(`    ${status} [${todo.id}] ${todo.text}`);
    });
});

// --- 演示 ---
console.log("\n--- 添加待办 ---");
store.dispatch(addTodo("学习 JavaScript"));
store.dispatch(addTodo("学习 TypeScript"));
store.dispatch(addTodo("学习 Node.js"));

console.log("\n--- 完成一项 ---");
const todoId = store.getState().todos[0].id;
store.dispatch(toggleTodo(todoId));

console.log("\n--- 筛选已完成 ---");
store.dispatch(setFilter("COMPLETED"));

console.log("\n--- 筛选未完成 ---");
store.dispatch(setFilter("ACTIVE"));

console.log("\n--- 显示全部 ---");
store.dispatch(setFilter("ALL"));

// --- 异步 Action ---
console.log("\n--- 异步 Action ---");

// 模拟异步获取数据
const fetchTodos = () => async (dispatch, getState) => {
    console.log("  开始获取远程数据...");

    // 模拟 API 调用
    await new Promise(r => setTimeout(r, 100));

    const remoteTodos = [
        { id: Date.now(), text: "远程任务 1", completed: false },
        { id: Date.now() + 1, text: "远程任务 2", completed: true }
    ];

    for (const todo of remoteTodos) {
        dispatch({
            type: ActionTypes.ADD_TODO,
            payload: todo
        });
    }

    console.log("  远程数据加载完成");
};

store.dispatch(fetchTodos());

// --- 时间旅行 ---
setTimeout(() => {
    console.log("\n--- 时间旅行 ---");
    console.log("执行撤销...");
    store.undo();

    console.log("\n执行重做...");
    store.redo();

    // 取消订阅
    unsubscribe();

    // --- 显示历史 ---
    console.log("\n--- 操作历史 ---");
    const history = store.getHistory();
    history.forEach((entry, i) => {
        console.log(`  ${i + 1}. ${entry.action.type}`);
    });

    console.log("\n【示例完成】");
}, 200);

// 导出
module.exports = {
    createStore,
    combineReducers,
    createAction,
    createReducer,
    createSelector,
    compose,
    // 中间件
    loggerMiddleware,
    thunkMiddleware,
    crashReporter
};
```
