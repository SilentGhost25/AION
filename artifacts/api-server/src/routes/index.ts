import { Router, type IRouter } from "express";
import healthRouter from "./health";
import materialsRouter from "./materials";
import generateRouter from "./generate";
import historyRouter from "./history";
import syllabusRouter from "./syllabus";

const router: IRouter = Router();

router.use(healthRouter);
router.use(materialsRouter);
router.use(generateRouter);
router.use(historyRouter);
router.use(syllabusRouter);

export default router;
