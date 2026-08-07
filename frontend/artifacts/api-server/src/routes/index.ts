import { Router, type IRouter } from "express";
import healthRouter from "./health";
import paperRouter from "./paper";
import questionRouter from "./question";

const router: IRouter = Router();

router.use(healthRouter);
router.use(paperRouter);
router.use(questionRouter);

export default router;
